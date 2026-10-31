import os
import sys
import time
import queue
import threading
import logging
from concurrent import futures
import grpc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 프로젝트 루트 경로 등록
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, now_dir)

import tts_pb2
import tts_pb2_grpc
from infer_tts_minimal import ZundamonTTSEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TTS_gRPC_Service")

# 전역 엔진 싱글톤 (서버 시작 시 1회 로드)
tts_engine: ZundamonTTSEngine = None


class TTSService(tts_pb2_grpc.TTSServiceServicer):
    """Gemini Live 등 실시간 대화형 AI 연동용 버퍼 큐 기반 gRPC 스트리밍 서비스"""

    def __init__(self, engine: ZundamonTTSEngine):
        self.engine = engine

    def StreamChatTTS(self, request_iterator, context):
        """
        양방향 스트리밍 핸들러 (Bidirectional Streaming):
        1. 클라이언트(Gemini Live 연결 서버)가 텍스트 청크를 쏟아내면 즉시 스레드 안전 큐(Queue)에 버퍼링합니다.
        2. TTS 합성 워커가 큐에서 문장을 순서대로 꺼내어 합성 후 오디오 PCM 바이트를 실시간 반환합니다.
        """
        buffer_queue = queue.Queue()
        sentinel = object()
        session_id = "default"

        def producer_worker():
            nonlocal session_id
            try:
                for req in request_iterator:
                    if req.session_id:
                        session_id = req.session_id
                    clean_text = req.text.strip()
                    if clean_text:
                        logger.info(f"📥 [수신 버퍼링] ({session_id}) '{clean_text}'")
                        buffer_queue.put(req)
                    if req.is_end_of_stream:
                        break
            except Exception as e:
                logger.error(f"🚨 요청 수신 중 에러: {e}")
            finally:
                buffer_queue.put(sentinel)

        # 텍스트 청크 수신 스레드 가동 (소비자 처리 속도와 무관하게 버퍼링)
        producer_thread = threading.Thread(target=producer_worker, daemon=True)
        producer_thread.start()

        # 소비자 루프 (큐에서 하나씩 꺼내어 순차 합성 및 스트리밍 yield)
        while True:
            item = buffer_queue.get()
            if item is sentinel:
                break

            text = item.text.strip()
            lang = item.text_lang or "ko"
            ref_name = item.ref_audio_name or "01_ref_emoNormal026.wav"
            speed = item.speed if item.speed > 0 else 1.0

            logger.info(f"🧠 [합성 시작] '{text}' (lang={lang}, ref={ref_name})")
            t0 = time.perf_counter()

            try:
                pcm_bytes = self.engine.generate_sentence_pcm(
                    text,
                    lang=lang,
                    ref_name=ref_name,
                    speed=speed,
                )
                elapsed = time.perf_counter() - t0
                logger.info(f"⚡ [합성 완료] '{text}' ({len(pcm_bytes)} bytes, {elapsed:.2f}초 소요)")

                if len(pcm_bytes) > 0:
                    yield tts_pb2.AudioChunkResponse(
                        session_id=session_id,
                        pcm_data=pcm_bytes,
                        sample_rate=self.engine.sampling_rate,
                        is_final=False,
                    )
            except Exception as e:
                logger.error(f"🚨 문장 합성 실패 ('{text}'): {e}")

        # 전체 스트림 완료 신호 전송
        yield tts_pb2.AudioChunkResponse(
            session_id=session_id,
            pcm_data=b"",
            sample_rate=self.engine.sampling_rate,
            is_final=True,
        )
        logger.info(f"🏁 [세션 스트림 종료] ({session_id})")

    def SynthesizeSentenceStream(self, request, context):
        """단일 문장 요청을 받아 작은 PCM 청크(1024 샘플 단위)로 쪼개어 스트리밍 반환"""
        text = request.text.strip()
        lang = request.text_lang or "ko"
        ref_name = request.ref_audio_name or "01_ref_emoNormal026.wav"
        speed = request.speed if request.speed > 0 else 1.0

        if not text:
            return

        pcm_bytes = self.engine.generate_sentence_pcm(text, lang=lang, ref_name=ref_name, speed=speed)
        chunk_size = 2048  # 1024 샘플 (16-bit Mono = 2바이트 * 1024)

        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            yield tts_pb2.AudioChunkResponse(
                session_id=request.session_id,
                pcm_data=chunk,
                sample_rate=self.engine.sampling_rate,
                is_final=(i + chunk_size >= len(pcm_bytes)),
            )

    def SynthesizeSentence(self, request, context):
        """단일 문장 요청에 대해 완성된 전체 PCM 오디오 일괄 반환 (Unary)"""
        text = request.text.strip()
        lang = request.text_lang or "ko"
        ref_name = request.ref_audio_name or "01_ref_emoNormal026.wav"
        speed = request.speed if request.speed > 0 else 1.0

        pcm_bytes = self.engine.generate_sentence_pcm(text, lang=lang, ref_name=ref_name, speed=speed)
        return tts_pb2.AudioChunkResponse(
            session_id=request.session_id,
            pcm_data=pcm_bytes,
            sample_rate=self.engine.sampling_rate,
            is_final=True,
        )


def serve(port: int = None):
    global tts_engine

    # Cloud Run 기본 포트 환경변수(PORT) 지원
    if port is None:
        port = int(os.environ.get("PORT", "50051"))

    logger.info("🚀 ZundamonTTSEngine 초기화 중...")
    tts_engine = ZundamonTTSEngine()
    
    logger.info("🔥 [Warm-up] ONNX Runtime / OpenVINO 그래프 컴파일 및 웜업 진행 중...")
    t_warm = time.perf_counter()
    tts_engine.warmup("안녕")
    logger.info(f"✨ [Warm-up 완료] 모든 신경망 그래프 컴파일 완료! ({time.perf_counter() - t_warm:.2f}초 소요)")
    logger.info("✅ ZundamonTTSEngine 로드 및 서빙 준비 완료!")

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        options=[
            ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ],
    )

    tts_pb2_grpc.add_TTSServiceServicer_to_server(TTSService(tts_engine), server)

    # Cloud Run 헬스체크 표준 프로브 대응 (grpc_health)
    try:
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc

        health_servicer = health.HealthServicer()
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        logger.info("🩺 gRPC 표준 Health Check 서비스 등록 완료")
    except ImportError:
        pass

    server_address = f"0.0.0.0:{port}"
    server.add_insecure_port(server_address)
    logger.info(f"🌟 gRPC TTS 스트리밍 서버가 가동되었습니다: {server_address}")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("🛑 서버 종료 중...")
        server.stop(0)


if __name__ == "__main__":
    serve()

