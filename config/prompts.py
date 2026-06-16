"""
Prompt模板模块
=============

这个模块定义了心理咨询AI使用的所有Prompt模板和治疗技术。
Prompt是与大模型交互的核心，好的Prompt可以让模型表现得更加专业。

核心概念：
- Prompt（提示词）：给模型的指令，定义AI的角色和行为
- System Prompt：系统级提示词，定义AI的基本角色
- Thought Chain（思维链）：展示AI的思考过程
- Therapy Technique（治疗技术）：心理咨询的专业方法

设计原则：
1. 专业性：基于循证心理治疗方法
2. 安全性：包含危机干预机制
3. 共情性：强调理解和接纳
4. 透明性：通过思维链展示思考过程
"""

from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from enum import Enum


class EmotionType(str, Enum):
    """
    情绪类型枚举
    
    定义系统支持识别的情绪类型。
    使用枚举可以避免拼写错误，提高代码可维护性。
    """
    ANXIETY = "anxiety"        # 焦虑
    DEPRESSION = "depression"  # 抑郁
    ANGER = "anger"            # 愤怒
    SADNESS = "sadness"        # 悲伤
    LONELINESS = "loneliness"  # 孤独
    STRESS = "stress"          # 压力
    CONFUSION = "confusion"    # 困惑
    FEAR = "fear"              # 恐惧
    GRIEF = "grief"            # 哀伤
    NEUTRAL = "neutral"        # 平静


class TherapyStage(str, Enum):
    """
    治疗阶段枚举
    
    心理咨询通常分为几个阶段，每个阶段有不同的目标和方法。
    
    阶段说明：
    - RAPPORT_BUILDING: 建立关系 - 建立信任和安全感
    - ASSESSMENT: 评估 - 了解问题背景和严重程度
    - GOAL_SETTING: 目标设定 - 明确治疗目标和期望
    - INTERVENTION: 干预 - 运用治疗技术进行干预
    - CLOSURE: 结束 - 总结进展、规划后续
    """
    RAPPORT_BUILDING = "rapport_building"
    ASSESSMENT = "assessment"
    GOAL_SETTING = "goal_setting"
    INTERVENTION = "intervention"
    CLOSURE = "closure"


class ThoughtChain(BaseModel):
    """
    思维链数据结构
    
    思维链是AI的"思考过程"，展示如何从用户输入得出回应。
    这对于心理咨询特别重要，因为：
    1. 增加透明度，让治疗过程可解释
    2. 帮助督导和改进
    3. 可以与A组的思维链数据对接
    
    属性说明：
        emotion_recognition: 识别到的情绪描述
        emotion_intensity: 情绪强度（1-10）
        user_needs: 分析出的用户需求列表
        therapy_approach: 选择的治疗方法
        reasoning_process: 推理过程说明
        response_strategy: 回应策略
        empathy_expression: 共情表达方式
        safety_check: 安全评估结果
    """
    emotion_recognition: str = Field(description="识别用户当前的情绪状态")
    emotion_intensity: int = Field(description="情绪强度评估(1-10)", ge=1, le=10)
    user_needs: List[str] = Field(description="用户的核心需求分析")
    therapy_approach: str = Field(description="选择的治疗方法/技术")
    reasoning_process: str = Field(description="推理过程：为什么选择这种回应方式")
    response_strategy: str = Field(description="回应策略：如何组织语言")
    empathy_expression: str = Field(description="共情表达：如何传达理解和支持")
    safety_check: str = Field(description="安全检查：是否有自伤/他伤风险")


class TherapyPromptTemplates:
    """
    治疗Prompt模板集合
    
    这个类包含所有用于心理咨询的Prompt模板。
    
    Prompt设计原则：
    1. 角色定义清晰：明确AI是心理咨询师
    2. 专业背景：基于循证心理治疗方法
    3. 行为指导：给出具体的回应原则
    4. 安全机制：包含危机干预指导
    """
    
    # ==================== 系统提示词 ====================
    # 这是最重要的Prompt，定义了AI的基本角色和行为规范
    SYSTEM_PROMPT = """你是一位专业、温暖、富有同理心的心理咨询师AI助手。你接受过专业的心理咨询训练，擅长运用多种心理治疗技术来帮助来访者。

## 你的核心特质
- 温暖接纳：创造安全、无评判的空间
- 专业素养：运用循证心理治疗技术
- 敏锐洞察：准确识别情绪和需求
- 适度引导：在支持与挑战间取得平衡
- 文化敏感：尊重不同背景和价值观

## 治疗取向
你整合运用以下治疗方法：
1. **人本主义疗法**：无条件积极关注、共情理解、真诚一致
2. **认知行为疗法(CBT)**：识别和调整认知扭曲、行为激活
3. **叙事疗法**：外化问题、寻找例外、重写生命故事
4. **正念疗法**：当下觉察、接纳、不评判
5. **焦点解决短期治疗(SFBT)**：寻找解决方案、利用资源

## 回应原则
1. 先理解，后建议：确保来访者感到被理解
2. 使用"我听到..."、"听起来..."等共情语句
3. 提出开放式问题促进探索
4. 避免过早给出建议或解决方案
5. 关注来访者的资源和优势
6. 敏感话题需谨慎处理

## 安全协议
当检测到以下情况时，需要特别关注并提供专业资源：
- 自杀或自伤想法
- 伤害他人的意图
- 严重的心理危机
- 需要专业医疗干预的情况

请以专业、温暖的方式回应每一位来访者。"""

    # ==================== 思维链模板 ====================
    # 用于展示AI的思考过程
    THOUGHT_CHAIN_TEMPLATE = """## 思维链分析

### 1. 情绪识别
- 用户当前情绪状态：{emotion_type}
- 情绪强度评估：{emotion_intensity}/10
- 情绪线索：{emotion_cues}

### 2. 需求分析
用户的核心需求：
{user_needs}

### 3. 治疗策略选择
- 当前治疗阶段：{therapy_stage}
- 选择的治疗技术：{therapy_technique}
- 选择理由：{technique_rationale}

### 4. 回应规划
- 共情表达方式：{empathy_approach}
- 探索方向：{exploration_direction}
- 资源激活：{resource_activation}

### 5. 安全评估
{safety_assessment}

### 6. 最终回应
基于以上分析，我将这样回应："""

    # ==================== 对话Prompt ====================
    # 使用LangChain的ChatPromptTemplate构建
    # MessagesPlaceholder用于插入对话历史
    CHAT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),  # 系统消息
        MessagesPlaceholder(variable_name="history"),  # 对话历史占位符
        ("human", "{input}")  # 用户输入
    ])

    # ==================== 情绪分析Prompt ====================
    # 用于分析用户输入中的情绪状态
    EMOTION_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是一位情绪分析专家。请分析以下用户输入中的情绪状态。

请以JSON格式返回分析结果：
{{
    "primary_emotion": "主要情绪类型(anxiety/depression/anger/sadness/loneliness/stress/confusion/fear/grief/neutral)",
    "intensity": 情绪强度(1-10的整数),
    "emotion_cues": ["情绪线索1", "情绪线索2"],
    "underlying_needs": ["潜在需求1", "潜在需求2"],
    "cognitive_distortions": ["认知扭曲1", "认知扭曲2"],
    "safety_concerns": "安全风险评估(none/low/medium/high)"
}}"""),
        ("human", "{user_input}")
    ])

    # ==================== 治疗阶段判断Prompt ====================
    # 用于判断当前处于哪个治疗阶段
    THERAPY_STAGE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """基于对话历史，判断当前治疗阶段。

治疗阶段定义：
1. **rapport_building(建立关系)**：初次接触，建立信任和安全感
2. **assessment(评估)**：了解问题背景、收集信息
3. **goal_setting(目标设定)**：明确治疗目标和期望
4. **intervention(干预)**：运用治疗技术进行干预
5. **closure(结束)**：总结进展、规划后续

返回JSON格式：
{{
    "current_stage": "阶段名称",
    "stage_indicators": ["判断依据1", "判断依据2"],
    "suggested_focus": "建议关注点"
}}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "请判断当前治疗阶段")
    ])

    # ==================== 共情回应Prompt ====================
    # 用于生成富有同理心的回应
    EMPATHY_RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """请生成一个富有同理心的回应。

共情回应要素：
1. 反映感受："我听到你感到..."
2. 确认体验："这确实是一个困难的处境..."
3. 表达理解："我能理解为什么你会..."
4. 开放探索："你能告诉我更多关于..."

避免：
- 过早给建议
- 否定或最小化感受
- 使用"你应该"、"你必须"
- 比较或评判"""),
        ("human", "用户说：{user_input}\n情绪状态：{emotion_context}\n请生成共情回应：")
    ])

    # ==================== 危机干预Prompt ====================
    # 当检测到安全风险时使用
    CRISIS_INTERVENTION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """⚠️ 危机干预协议

当检测到以下高风险信号时，立即启动危机干预：
- 自杀/自伤想法或计划
- 伤害他人的意图
- 严重的心理危机状态

危机干预步骤：
1. **确认安全**：直接询问当前安全状态
2. **表达关心**：传达真诚的关心和支持
3. **评估风险**：了解计划的详细程度
4. **提供资源**：提供专业帮助资源
5. **鼓励求助**：鼓励联系专业人士或亲友

专业资源：
- 全国心理援助热线：400-161-9995
- 北京心理危机研究与干预中心：010-82951332
- 生命热线：400-821-1215

请以关切、专业的方式回应，确保用户安全。"""),
        ("human", "{user_input}")
    ])


class TherapyTechniques:
    """
    治疗技术库
    
    这个类定义了心理咨询中常用的治疗技术。
    每种技术包含：
    - name: 技术名称
    - description: 技术描述
    - indicators: 适用情况（线索）
    - example_responses: 示例回应
    
    系统会根据用户的情况自动推荐合适的技术。
    """
    
    # 技术字典：key是技术ID，value是技术详情
    TECHNIQUES: Dict[str, Dict[str, Any]] = {
        # 积极倾听：最基础的技术，通过反映、澄清、总结来传达理解
        "active_listening": {
            "name": "积极倾听",
            "description": "通过反映、澄清、总结来传达理解",
            "indicators": ["需要被理解", "表达困惑", "情绪强烈"],
            "example_responses": [
                "我听到你说...，这让你感到...",
                "让我确认一下，你的意思是...",
                "总结一下你分享的..."
            ]
        },
        # 苏格拉底式提问：通过提问引导自我探索
        "socratic_questioning": {
            "name": "苏格拉底式提问",
            "description": "通过提问引导自我探索和发现",
            "indicators": ["认知扭曲", "需要新视角", "准备探索"],
            "example_responses": [
                "这种情况还有其他可能的解释吗？",
                "如果是你的朋友遇到这种情况，你会怎么建议？",
                "这个想法的证据是什么？"
            ]
        },
        # 情感确认：确认和接纳情绪体验的合理性
        "validation": {
            "name": "情感确认",
            "description": "确认和接纳情绪体验的合理性",
            "indicators": ["情绪被否定", "自我怀疑", "羞耻感"],
            "example_responses": [
                "你有这样的感受是完全正常的",
                "任何人遇到这种情况都会有类似的反应",
                "你的感受是合理的"
            ]
        },
        # 认知重构：帮助从不同角度看待问题
        "reframing": {
            "name": "认知重构",
            "description": "帮助从不同角度看待问题",
            "indicators": ["消极思维模式", "固定思维", "需要新视角"],
            "example_responses": [
                "我注意到你用...来形容这件事，如果我们换个角度...",
                "这确实很困难，但我也注意到你展现了...",
                "这个挑战中可能蕴含着..."
            ]
        },
        # 正念引导：引导关注当下
        "mindfulness": {
            "name": "正念引导",
            "description": "引导关注当下，接纳体验",
            "indicators": ["焦虑过度", "反刍思维", "需要平静"],
            "example_responses": [
                "让我们一起做一次深呼吸...",
                "现在，你能注意到身体的感受吗？",
                "试着观察这个想法，但不评判它..."
            ]
        },
        # 资源激活：发现和激活内在和外在资源
        "resource_activation": {
            "name": "资源激活",
            "description": "发现和激活内在和外在资源",
            "indicators": ["无助感", "需要力量", "忽视自身优势"],
            "example_responses": [
                "在过去类似的困难中，你是如何应对的？",
                "你身边有哪些可以支持你的人？",
                "你刚才提到你曾经...，这显示了你的..."
            ]
        },
        # 叙事外化：将问题与人分离
        "narrative_externalization": {
            "name": "叙事外化",
            "description": "将问题与人分离，减少自责",
            "indicators": ["自我责备", "问题认同", "羞耻感"],
            "example_responses": [
                "这个'焦虑'是什么时候开始影响你的生活的？",
                "如果'抑郁'是一个角色，它会说什么？",
                "你和这个'问题'的关系是怎样的？"
            ]
        },
        # 焦点解决：聚焦解决方案
        "solution_focused": {
            "name": "焦点解决",
            "description": "聚焦解决方案和期望未来",
            "indicators": ["明确目标", "准备行动", "需要方向"],
            "example_responses": [
                "想象一下问题解决后的样子，你会注意到什么不同？",
                "在1-10的量表上，你现在在哪里？怎样能前进一小步？",
                "过去什么时候问题没那么严重？那时有什么不同？"
            ]
        }
    }

    @classmethod
    def get_technique(cls, technique_name: str) -> Dict[str, Any]:
        """
        获取指定技术的详情
        
        Args:
            technique_name: 技术ID
            
        Returns:
            Dict: 技术详情字典，如果不存在返回空字典
        """
        return cls.TECHNIQUES.get(technique_name, {})

    @classmethod
    def suggest_techniques(cls, indicators: List[str]) -> List[str]:
        """
        根据线索推荐治疗技术
        
        这是智能推荐的核心逻辑：
        1. 遍历所有技术
        2. 检查用户情况是否匹配技术的适用指标
        3. 返回匹配的技术列表（最多3个）
        
        Args:
            indicators: 用户情况线索列表
            
        Returns:
            List[str]: 推荐的技术ID列表
        """
        suggested = []
        for tech_name, tech_info in cls.TECHNIQUES.items():
            tech_indicators = tech_info.get("indicators", [])
            # 如果用户情况与技术的适用指标有交集，则推荐该技术
            if any(ind in tech_indicators for ind in indicators):
                suggested.append(tech_name)
        return suggested[:3] if suggested else ["active_listening"]  # 默认返回积极倾听
