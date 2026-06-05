from __future__ import annotations

import base64

from llm_client import get_client

DEFAULT_SYSTEM_PROMPT = (
    "你是 DisasterClaw 的视觉分析模块。"
    "你需要分析上传的灾害现场图片，并用中文给出简洁、可执行的判断。"
    "回答尽量包含：场景摘要、关键风险、可行动目标、建议动作。"
    "如果图像信息不足，要明确说明不确定性。"
)

DEFAULT_USER_PROMPT = "请分析这张图片中的灾害迹象、风险点、可通行区域和值得标记的目标。"


class VLMAnalyzer:
    def __init__(self):
        self._client = get_client(module="vlm")

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def provider_url(self) -> str:
        return self._client.provider_url

    def analyze_image_bytes(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        prompt: str = "",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 700,
    ) -> dict:
        if not image_bytes:
            raise ValueError("empty image payload")

        data_url = _build_data_url(image_bytes, mime_type)
        user_prompt = prompt.strip() or DEFAULT_USER_PROMPT
        styles = _message_style_order(self._client)
        last_error: Exception | None = None

        for style in styles:
            try:
                text = self._client.chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                _build_image_part(data_url, style),
                            ],
                        },
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                return {
                    "analysis": text.strip(),
                    "model": self.model,
                    "provider_url": self.provider_url,
                    "image_input_mode": style,
                }
            except Exception as exc:
                last_error = exc
                # 连接错误通常不是格式问题，不再重试第二种消息格式。
                if "无法连接" in str(exc):
                    break

        assert last_error is not None
        raise last_error


def _build_data_url(image_bytes: bytes, mime_type: str) -> str:
    payload = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _message_style_order(client) -> list[str]:
    api_type = str(client.provider_option("api_type", "") or "").strip().lower()
    if api_type == "local_qwen_vl":
        return ["local"]

    configured = str(client.provider_option("image_input_mode", "auto") or "auto").strip().lower()
    if configured in {"openai", "ollama"}:
        return [configured]

    url = client.provider_url.lower()
    if "11434" in url or "ollama" in url:
        return ["ollama", "openai"]
    return ["openai", "ollama"]


def _build_image_part(data_url: str, style: str) -> dict:
    if style == "ollama":
        return {"type": "image_url", "image_url": data_url}
    if style == "local":
        return {"type": "image_url", "image_url": {"url": data_url}}
    return {"type": "image_url", "image_url": {"url": data_url}}
