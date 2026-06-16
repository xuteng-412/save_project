import asyncio
import os
import sys
import uuid

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.llm import get_llm_adapter, LLMConfig
from core.memory import TherapySessionMemory, SessionManager
from core.chain import TherapyChain
from config.settings import settings
from utils import format_thought_chain_output, format_emotion_analysis, get_timestamp


class InteractiveTherapySession:
    def __init__(self, use_mock: bool = True):
        self.session_id = str(uuid.uuid4())[:8]
        self.use_mock = use_mock

        if use_mock:
            self.llm = get_llm_adapter("mock")
            print("使用Mock模型进行测试")
        else:
            self.llm = get_llm_adapter("qwen")
            print(f"连接模型: {settings.MODEL_NAME}")
            print(f"API地址: {settings.OPENAI_API_BASE}")

        self.session = SessionManager.get_session(self.session_id)
        self.chain = TherapyChain(
            llm_adapter=self.llm,
            enable_thought_chain=True,
            enable_emotion_analysis=True
        )

    async def chat(self, user_input: str) -> None:
        print(f"\n[{get_timestamp()}] 用户: {user_input}")

        response = await self.chain.chat(user_input, self.session)

        if response.safety_alert:
            print("\n⚠️ 安全警报已触发")

        if response.emotion_analysis:
            print(format_emotion_analysis({
                "primary_emotion": response.emotion_analysis.primary_emotion,
                "intensity": response.emotion_analysis.intensity,
                "emotion_cues": response.emotion_analysis.emotion_cues,
                "underlying_needs": response.emotion_analysis.underlying_needs,
                "cognitive_distortions": response.emotion_analysis.cognitive_distortions,
                "safety_concerns": response.emotion_analysis.safety_concerns
            }))

        if response.thought_chain:
            print(format_thought_chain_output({
                "emotion_recognition": response.thought_chain.emotion_recognition,
                "emotion_intensity": response.thought_chain.emotion_intensity,
                "user_needs": response.thought_chain.user_needs,
                "therapy_approach": response.thought_chain.therapy_approach,
                "reasoning_process": response.thought_chain.reasoning_process,
                "response_strategy": response.thought_chain.response_strategy,
                "empathy_expression": response.thought_chain.empathy_expression,
                "safety_check": response.thought_chain.safety_check
            }))

        if response.suggested_techniques:
            print(f"\n建议技术: {', '.join(response.suggested_techniques)}")

        print(f"\n[{get_timestamp()}] AI咨询师: {response.response}")
        print(f"\n当前治疗阶段: {response.therapy_stage}")

    def show_session_info(self) -> None:
        trend = self.session.get_emotion_trend()
        print("\n" + "=" * 50)
        print("会话信息")
        print("=" * 50)
        print(f"会话ID: {self.session_id}")
        print(f"消息数量: {self.session.metadata.message_count}")
        print(f"治疗阶段: {self.session.metadata.therapy_stage}")
        print(f"情绪趋势: {trend['trend']}")
        print(f"平均情绪强度: {trend['average_intensity']:.1f}")
        if self.session.metadata.key_topics:
            print(f"关键话题: {', '.join(self.session.metadata.key_topics)}")


async def main():
    print("\n" + "=" * 60)
    print("心理咨询AI助手 - 交互式测试")
    print("=" * 60)
    print("\n命令说明:")
    print("  - 直接输入文字进行对话")
    print("  - 输入 'info' 查看会话信息")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'new' 开始新会话")
    print("=" * 60)

    use_mock = input("\n是否使用Mock模型测试? (y/n, 默认y): ").strip().lower()
    use_mock = use_mock != 'n'

    session = InteractiveTherapySession(use_mock=use_mock)

    print(f"\n新会话已创建，ID: {session.session_id}")
    print("开始对话吧！（输入 'quit' 退出）")

    while True:
        try:
            user_input = input("\n你: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                session.show_session_info()
                print("\n感谢使用，再见！")
                break

            if user_input.lower() == 'info':
                session.show_session_info()
                continue

            if user_input.lower() == 'new':
                session = InteractiveTherapySession(use_mock=use_mock)
                print(f"\n新会话已创建，ID: {session.session_id}")
                continue

            await session.chat(user_input)

        except KeyboardInterrupt:
            print("\n\n会话已中断")
            session.show_session_info()
            break
        except Exception as e:
            print(f"\n错误: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
