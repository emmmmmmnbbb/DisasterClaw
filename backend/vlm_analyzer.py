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

# Agent-VQA 结构化问答 system prompt (计划 7.2)。强制 JSON schema, 封闭答案集合。
AGENT_VQA_SYSTEM_PROMPT = (
    "你是 DisasterClaw 的灾情视觉问答模块。根据当前俯视观测图像与结构化感知证据，"
    "用中文回答灾情问题。必须输出严格 JSON，字段：answer（字符串，来自题目给定选项）、"
    "confidence（0-1 浮点）、abstain（布尔）、decision（answer|continue_search|reobserve|abstain）、"
    "reason_code（sufficient_evidence|target_missing|low_confidence|budget_exhausted）、"
    "evidence（对象，必须含 source）。source 只能逐字填写英文枚举 image、detector、"
    "change_classifier、semantic_map、history 之一，绝不能填写证据描述句；若只有图像"
    "就填 image。可把简短证据描述放在 evidence.note。定位到目标时给 norm_xy=[x,y]，"
    "采用左上(0,0)、右下(1,1)的图像坐标；未定位时不要输出 norm_xy。"
    "若题目提到视场中心十字标记建筑，只判断该十字所指建筑。"
    "decision 与 reason_code 必须成对，不得混用：decision=answer 时 reason_code 只能是"
    "sufficient_evidence；decision=continue_search 时只能是 target_missing；"
    "decision=reobserve 时只能是 low_confidence；decision=abstain 时可用其余任一原因码。"
    "若画面中确实找不到题目所指的目标、无法确定其方位或等级，必须选 continue_search"
    "或 abstain，禁止一边选 answer 一边把 reason_code 写成 target_missing 或"
    "low_confidence 来蒙混过关——这种输出会被判为非法并整题作废。"
    "作答格式示例：{\"answer\":\"是\",\"confidence\":0.8,\"abstain\":false,"
    "\"decision\":\"answer\",\"reason_code\":\"sufficient_evidence\","
    "\"evidence\":{\"source\":\"image\",\"note\":\"简短证据\"}}。"
    "不得输出思维链或 JSON 以外的自由散文。信息不足时 abstain=true，不得猜测；"
    "即使弃答，answer 也必须填写一个给定候选项作为未采纳的候选答案。"
)


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

    def answer_image_question(
        self,
        image_bytes: bytes,
        question: str,
        choices: list[str] | None = None,
        evidence_text: str = "",
        system_prompt: str = AGENT_VQA_SYSTEM_PROMPT,
        max_tokens: int = 300,
        temperature: float = 0.1,
    ) -> dict:
        """Agent-VQA 结构化问答接口 (计划 7.2)。

        返回 dict（非已解析的 VqaAnswer）：
          - raw: VLM 原始文本（供 parse_vlm_json_output 二次解析与缓存）
          - model / provider_url / image_input_mode: 运行元数据
          - prompt: 实际下发的 user prompt（含问题、选项、证据，供 manifest 记录）

        解析失败时由调用方（AgentVqaController）经 parse_vlm_json_output 转成
        显式 invalid_output，本接口不静默猜测。
        """
        if not image_bytes:
            raise ValueError("empty image payload")
        data_url = _build_data_url(image_bytes, "image/jpeg")
        parts = [question]
        if choices:
            parts.append("候选答案：" + " / ".join(choices))
        if evidence_text:
            parts.append("结构化证据：" + evidence_text)
            parts.append("本题包含结构化证据；evidence.source 仍只能填写规定的英文枚举。")
        else:
            parts.append('本题只有图像证据；evidence.source 必须逐字填写 "image"。')
        parts.append("请只输出单个 JSON 对象，不要输出 Markdown 代码围栏或解释。")
        user_prompt = "\n".join(parts)
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
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return {
                    "raw": text.strip(),
                    "model": self.model,
                    "provider_url": self.provider_url,
                    "image_input_mode": style,
                    "prompt": user_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            except Exception as exc:
                last_error = exc
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
