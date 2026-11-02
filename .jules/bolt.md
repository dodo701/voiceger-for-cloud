## 2024-11-01 - ONNX Runtime IOBinding Ping-Pong Strategy
**Learning:** When using ONNX Runtime for auto-regressive decoding, continuously binding inputs using `sess.run()` causes severe memory overhead and stalls due to frequent C++/Python memory transfers.
**Action:** Use `sess.run_with_iobinding()` instead. To prevent memory corruption or use-after-free errors when rapidly alternating bindings, use a "ping-pong" buffer strategy with two `io_binding()` instances, and feed inputs using `ort.OrtValue.ortvalue_from_numpy()` to keep memory entirely on the C++ side.
