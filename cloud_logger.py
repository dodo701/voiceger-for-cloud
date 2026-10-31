"""
Google Cloud Logging 및 Error Reporting 통합 모듈
- Google Cloud (Cloud Run / GKE / Compute Engine):
  1) google-cloud-logging: 구조화된 JSON 로그(Structured Logging) 및 Cloud Logging Log Explorer 연동
  2) google-cloud-error-reporting: logging.ERROR 및 예외(Traceback) 자동 감지하여 GCP Error Reporting 대시보드로 실시간 리포팅
- 로컬 개발 환경: 지연(Timeout) 없는 즉각적인 컬러 콘솔 포맷터(Color/Timestamp)로 자동 폴백
- 기존 코드 호환성: 기존의 logger.error(..., exc_info=True) 및 logger.exception(...) 호출이 아무런 코드 수정 없이 자동으로 Error Reporting에 연동됩니다.
"""
import os
import sys
import logging
from typing import Optional, Dict, Any

# Windows 콘솔 cp949 유니코드 인코딩 에러 원천 방지
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_cloud_logging_initialized = False
_error_client = None


class ColoredConsoleFormatter(logging.Formatter):
    """로컬 콘솔 디버깅용 가독성 포맷터"""
    COLOR_CODES = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLOR_CODES.get(record.levelno, self.RESET)
        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        prefix = f"{color}[{record.levelname}]{self.RESET}"
        return f"{time_str} {prefix} [{record.name}] {record.getMessage()}"


class GCPErrorReportingLoggingHandler(logging.Handler):
    """
    Python logging과 Google Cloud Error Reporting을 연결하는 브릿지 핸들러.
    루트 로거에 부착되어 logger.error()나 logger.exception()이 발생하면
    스택 트레이스와 함께 GCP Error Reporting 콘솔로 자동 전송합니다.
    """
    def __init__(self, error_client, level=logging.ERROR):
        super().__init__(level=level)
        self.error_client = error_client

    def emit(self, record: logging.LogRecord):
        if self.error_client is None:
            return
        try:
            if record.exc_info:
                # traceback이 포함된 경우 (logger.exception 또는 exc_info=True)
                self.error_client.report_exception(
                    additional_message=record.getMessage()
                )
            else:
                # 일반 ERROR/CRITICAL 로그
                self.error_client.report(record.getMessage())
        except Exception:
            # 로깅 과정 중 발생한 오류로 인해 메인 서비스가 중단되지 않도록 보호
            self.handleError(record)


def is_gcp_environment() -> bool:
    """Cloud Run, Cloud Functions, GKE, 또는 명시적 GCP 자격증명이 있는 환경인지 판별합니다."""
    gcp_indicators = [
        "K_SERVICE",                   # Cloud Run 기본 환경변수
        "FUNCTION_NAME",               # Cloud Functions
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GCP_PROJECT",
    ]
    return any(os.environ.get(k) for k in gcp_indicators)


def setup_cloud_logging(service_name: str = "voiceger-tts", log_level: int = logging.INFO) -> logging.Logger:
    """
    Google Cloud Logging 및 Error Reporting 핸들러를 구성하고 애플리케이션 루트 로거를 설정합니다.
    - GCP 환경(Cloud Run 등):
      1. Cloud Logging 클라이언트로 모든 로그 수집
      2. Error Reporting 핸들러로 logger.error / 예외 자동 추적
      3. 컨테이너 콘솔(stdout) 출력 유지
    - 로컬 환경: 메타데이터 서버 타임아웃 없이 즉시 컬러 콘솔 로거로 기동
    """
    global _cloud_logging_initialized, _error_client

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not _cloud_logging_initialized:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredConsoleFormatter())
        console_handler.setLevel(log_level)

        # 1. GCP 환경 감지 시 Cloud Logging 및 Error Reporting 클라이언트 연동
        if is_gcp_environment():
            # A. Cloud Logging
            try:
                import google.cloud.logging
                logging_client = google.cloud.logging.Client()
                logging_client.setup_logging(log_level=log_level)
                root_logger.addHandler(console_handler)
                root_logger.info(f"☁️ [Cloud Logging] Google Cloud Logging 활성화 완료 (Service: {service_name})")
            except Exception as e:
                root_logger.addHandler(console_handler)
                root_logger.warning(f"⚠️ [Cloud Logging] GCP 감지되었으나 Cloud Logging 연결 실패, 콘솔 유지: {e}")

            # B. Error Reporting
            try:
                from google.cloud import error_reporting
                version = os.environ.get("K_REVISION") or os.environ.get("APP_VERSION", "v1")
                _error_client = error_reporting.Client(service=service_name, version=version)
                error_handler = GCPErrorReportingLoggingHandler(_error_client, level=logging.ERROR)
                root_logger.addHandler(error_handler)
                root_logger.info(f"🚨 [Error Reporting] Google Cloud Error Reporting 활성화 완료 (Service: {service_name}, Version: {version})")
            except Exception as e:
                root_logger.warning(f"⚠️ [Error Reporting] Error Reporting 연결 실패: {e}")
        else:
            # 2. 로컬 개발 환경: 지연 없이 즉각 콘솔 핸들러 적용
            root_logger.handlers = [h for h in root_logger.handlers if not isinstance(h, logging.StreamHandler)]
            root_logger.addHandler(console_handler)
            root_logger.info("💻 [Local Logger] 로컬 환경이 감지되어 고속 컬러 콘솔 로거로 기동합니다. (Error Reporting은 로컬 콘솔에만 출력)")

        _cloud_logging_initialized = True

    return logging.getLogger(service_name)


def report_exception(e: Optional[Exception] = None, message: Optional[str] = None):
    """
    필요 시 try/except 블록 안에서 명시적으로 Error Reporting에 오류를 보고할 때 사용하는 헬퍼.
    로컬 환경에서는 콘솔에 ERROR 레벨로 출력됩니다.
    """
    global _error_client
    logger = logging.getLogger("voiceger-tts")
    if _error_client is not None:
        try:
            _error_client.report_exception(additional_message=message)
            return
        except Exception:
            pass

    # 폴백: 표준 로거를 통한 출력
    if message:
        logger.error(f"{message}: {e}" if e else message, exc_info=True)
    else:
        logger.error(str(e) if e else "Unknown exception", exc_info=True)


def log_latency(logger: logging.Logger, session_id: str, text: str, latency: Dict[str, Any]):
    """
    청크별 세부 파이프라인 소요 시간 및 RTF 메트릭을 정형화하여 기록합니다.
    Google Cloud Logging Log Explorer에서 JSON 필드로 인덱싱 및 쿼리 가능합니다.
    """
    text_preview = text.strip()
    if len(text_preview) > 30:
        text_preview = text_preview[:30] + "..."

    msg = (
        f"⚡ [Latency] session={session_id} | "
        f"G2P={latency.get('text_frontend_ms', 0):.1f}ms | "
        f"GPT={latency.get('gpt_latent_ms', 0):.1f}ms | "
        f"VITS={latency.get('vits_decode_ms', 0):.1f}ms | "
        f"Total={latency.get('total_elapsed_ms', 0):.1f}ms | "
        f"Audio={latency.get('audio_duration_sec', 0):.2f}s | "
        f"RTF={latency.get('rtf', 0):.2f}x | "
        f"Text='{text_preview}'"
    )
    extra_payload = {
        "session_id": session_id,
        "text_preview": text_preview,
        **latency,
    }
    logger.info(msg, extra={"json_fields": extra_payload})
