import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import grpc
import soundfile as sf
import numpy as np

# 프로젝트 루트 경로 등록
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, now_dir)

import tts_pb2
import tts_pb2_grpc


def run_grpc_test():
    channel = grpc.insecure_channel("localhost:50051")
    stub = tts_pb2_grpc.TTSServiceStub(channel)

    # Gemini Live에서 들어온다고 가정한 2개의 연속 문장 청크
    sentences = [
        "야호! 나는 즌다몬인 거시다!",
        "오늘 날씨가 정말 맑고 상쾌한 거시다!",
    ]

    def request_generator():
        for i, text in enumerate(sentences):
            print(f"📤 [클라이언트 송신] 청크 {i+1}: '{text}'")
            yield tts_pb2.TextChunkRequest(
                session_id="gemini-live-test",
                text=text,
                text_lang="ko",
                ref_audio_name="01_ref_emoNormal026.wav",
                speed=1.0,
                is_end_of_stream=(i == len(sentences) - 1),
            )
            time.sleep(0.1)  # LLM이 문장을 쏘아 올리는 속도 흉내

    print("🚀 [gRPC 클라이언트] StreamChatTTS 양방향 스트리밍 요청 시작...")
    response_stream = stub.StreamChatTTS(request_generator())

    received_pcm_list = []
    chunk_count = 0

    for resp in response_stream:
        if resp.is_final:
            print("🏁 [클라이언트 수신] 전체 스트림 종료 신호 수신!")
            break

        chunk_count += 1
        print(f"📥 [클라이언트 수신] 오디오 청크 {chunk_count}: {len(resp.pcm_data)} bytes (SR: {resp.sample_rate})")
        received_pcm_list.append(resp.pcm_data)

    if received_pcm_list:
        full_pcm_bytes = b"".join(received_pcm_list)
        # 16-bit Mono PCM을 numpy 배열로 복원하여 WAV 파일로 저장
        audio_int16 = np.frombuffer(full_pcm_bytes, dtype=np.int16)
        output_file = "grpc_stream_output.wav"
        sf.write(output_file, audio_int16, 32000)
        print(f"🎉 [성공] 스트리밍 수신된 전체 오디오가 '{output_file}'로 저장되었습니다! ({len(audio_int16)} 샘플)")
    else:
        print("⚠️ 오디오 수신 실패")


if __name__ == "__main__":
    run_grpc_test()
