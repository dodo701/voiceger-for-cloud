import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_sdec_model(
    input_path: str = "GPT_weights_v2/zundamon_t2s_sdec.onnx",
    output_path: str = "GPT_weights_v2/zundamon_t2s_sdec_int8.onnx",
):
    print("=" * 60)
    print("🚀 GPT T2S Stage Decoder (sdec) 8비트 동적 양자화 시작")
    print(f"📥 입력 모델: {input_path}")
    print(f"📤 출력 모델: {output_path}")
    print("=" * 60)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 모델 파일을 찾을 수 없습니다: {input_path}")

    in_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"📊 원본 모델 크기: {in_size_mb:.2f} MB")

    t0 = time.time()
    print("⏳ MatMul 연산자 대상 per-channel INT8 가중치 양자화 진행 중...")

    # 동적 양자화(Dynamic Quantization):
    # - 활성화(Activation) 값은 float32를 유지하며 런타임에 동적 스케일링
    # - 가중치(Weight)만 8비트(QInt8)로 압축하여 정확도 손실을 최소화
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        weight_type=QuantType.QInt8,
    )

    t1 = time.time()
    out_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1.0 - (out_size_mb / in_size_mb)) * 100

    print(f"✅ 양자화 완료! (소요 시간: {t1 - t0:.2f}초)")
    print(f"📦 양자화 모델 크기: {out_size_mb:.2f} MB (용량 절감률: {reduction:.1f}%)")

    # 1. ONNX 모델 구조 무결성 검증
    print("🔍 ONNX 모델 구조 검증(check_model) 중...")
    model_proto = onnx.load(output_path)
    onnx.checker.check_model(model_proto)
    print("✅ ONNX 모델 구조 검증 통과!")

    # 2. ONNX Runtime 세션 로드 및 I/O 일치 검증
    print("🔍 ONNX Runtime 세션 로드 및 I/O 시그니처 검증 중...")
    orig_sess = ort.InferenceSession(input_path, providers=["CPUExecutionProvider"])
    quant_sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

    orig_inputs = [i.name for i in orig_sess.get_inputs()]
    quant_inputs = [i.name for i in quant_sess.get_inputs()]
    orig_outputs = [o.name for o in orig_sess.get_outputs()]
    quant_outputs = [o.name for o in quant_sess.get_outputs()]

    assert orig_inputs == quant_inputs, f"입력 불일치: {orig_inputs} != {quant_inputs}"
    assert orig_outputs == quant_outputs, f"출력 불일치: {orig_outputs} != {quant_outputs}"

    print(f"✅ 입력 텐서 검증 완료: {quant_inputs}")
    print(f"✅ 출력 텐서 검증 완료: {quant_outputs}")
    print("=" * 60)
    print("🎉 8비트 양자화 모델이 정상적으로 생성 및 검증되었습니다!")
    print(f"👉 사용 경로: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    now_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(now_dir, "GPT_weights_v2", "zundamon_t2s_sdec.onnx")
    output_file = os.path.join(now_dir, "GPT_weights_v2", "zundamon_t2s_sdec_int8.onnx")
    quantize_sdec_model(input_file, output_file)

