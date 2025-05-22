# 实践案例对比

本文档通过具体的实践案例来展示 LangChain 和 LangGraph 在不同场景下的应用差异。

## 案例1：简单问答机器人

### 需求描述
构建一个简单的客服机器人，能够回答常见问题，支持基础的多轮对话。

### LangChain 实现

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# 定义提示模板
template = """你是一个友好的客服助手。请根据对话历史回答用户问题。

对话历史:
{history}

用户: {input}
助手:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# 创建对话链
llm = ChatOpenAI(temperature=0.7)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

# 使用示例
while True:
    user_input = input("用户: ")
    if user_input.lower() == 'quit':
        break
    
    response = conversation.predict(input=user_input)
    print(f"助手: {response}")
```

**优势：**
- 实现简单，代码量少
- 快速部署和迭代
- 适合原型验证

**局限：**
- 功能相对固定
- 难以添加复杂逻辑
- 缺乏持久化支持

### LangGraph 实现

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], "对话消息"]
    user_id: str
    context: dict

def chatbot_node(state: ConversationState):
    """主聊天节点"""
    llm = ChatOpenAI(temperature=0.7)
    
    system_prompt = """你是一个友好的客服助手。
    请根据对话历史和用户上下文回答问题。"""
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }

def should_continue(state: ConversationState):
    """决定是否继续对话"""
    last_message = state["messages"][-1]
    if "再见" in last_message.content or "结束" in last_message.content:
        return END
    return "chatbot"

# 构建图
workflow = StateGraph(ConversationState)
workflow.add_node("chatbot", chatbot_node)
workflow.set_entry_point("chatbot")
workflow.add_conditional_edges("chatbot", should_continue)

# 配置持久化
checkpointer = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=checkpointer)

# 使用示例
config = {"configurable": {"thread_id": "user-123"}}

while True:
    user_input = input("用户: ")
    if user_input.lower() == 'quit':
        break
    
    state = {
        "messages": [HumanMessage(content=user_input)],
        "user_id": "123",
        "context": {}
    }
    
    result = app.invoke(state, config=config)
    print(f"助手: {result['messages'][-1].content}")
```

**优势：**
- 支持复杂的对话流程控制
- 内置持久化，支持跨会话
- 可扩展性强，易于添加新功能

**考虑：**
- 代码复杂度相对较高
- 适合需要长期维护的项目

---

## 案例2：智能客服工单系统

### 需求描述
构建一个智能客服系统，能够：
1. 分析用户问题类型
2. 自动查询知识库
3. 复杂问题转人工处理
4. 生成工单并跟踪状态

### LangChain 实现（简化版）

```python
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 问题分类链
classify_prompt = PromptTemplate(
    input_variables=["question"],
    template="分析以下问题的类型: {question}\n类型:"
)

# 知识库查询链
search_prompt = PromptTemplate(
    input_variables=["question", "category"],
    template="根据问题类型 {category}，为问题 {question} 查询相关信息:"
)

# 回答生成链
answer_prompt = PromptTemplate(
    input_variables=["question", "search_result"],
    template="基于查询结果 {search_result}，回答问题 {question}:"
)

llm = ChatOpenAI()

# 创建顺序链
classify_chain = LLMChain(llm=llm, prompt=classify_prompt, output_key="category")
search_chain = LLMChain(llm=llm, prompt=search_prompt, output_key="search_result")
answer_chain = LLMChain(llm=llm, prompt=answer_prompt, output_key="answer")

overall_chain = SequentialChain(
    chains=[classify_chain, search_chain, answer_chain],
    input_variables=["question"],
    output_variables=["category", "search_result", "answer"]
)

# 使用示例
question = "我的订单状态怎么查询？"
result = overall_chain.invoke({"question": question})
print(result["answer"])
```

**局限性：**
- 无法处理需要人工干预的复杂场景
- 缺乏工单管理功能
- 无法实现动态路由

### LangGraph 实现（完整版）

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal
import uuid

class TicketState(TypedDict):
    question: str
    category: str
    priority: Literal["low", "medium", "high"]
    ticket_id: str
    knowledge_result: str
    answer: str
    needs_human: bool
    status: Literal["open", "in_progress", "resolved", "escalated"]

def classify_question(state: TicketState):
    """问题分类节点"""
    llm = ChatOpenAI()
    
    prompt = f"""
    分析以下客户问题，返回分类和优先级：
    问题: {state['question']}
    
    请以JSON格式返回：
    {{"category": "类别", "priority": "优先级(low/medium/high)"}}
    """
    
    response = llm.invoke(prompt)
    # 解析响应（简化处理）
    if "账单" in state['question'] or "付款" in state['question']:
        category = "billing"
        priority = "high"
    elif "技术" in state['question'] or "故障" in state['question']:
        category = "technical"
        priority = "medium"
    else:
        category = "general"
        priority = "low"
    
    return {
        "category": category,
        "priority": priority,
        "ticket_id": str(uuid.uuid4())[:8],
        "status": "open"
    }

def search_knowledge_base(state: TicketState):
    """知识库查询节点"""
    # 模拟知识库查询
    knowledge_base = {
        "billing": "账单相关问题请查看账户页面或联系财务部门",
        "technical": "技术问题请先尝试重启设备，如仍有问题请提供错误日志",
        "general": "一般问题请参考用户手册或FAQ页面"
    }
    
    result = knowledge_base.get(state["category"], "未找到相关信息")
    return {"knowledge_result": result}

def generate_answer(state: TicketState):
    """生成回答节点"""
    llm = ChatOpenAI()
    
    prompt = f"""
    基于以下信息为客户生成回答：
    问题: {state['question']}
    类别: {state['category']}
    知识库结果: {state['knowledge_result']}
    
    请生成专业、友好的回答。如果知识库信息不足以解决问题，请建议转人工处理。
    """
    
    response = llm.invoke(prompt)
    
    # 判断是否需要人工处理
    needs_human = (
        state["priority"] == "high" or 
        "转人工" in response.content or
        "联系" in response.content
    )
    
    return {
        "answer": response.content,
        "needs_human": needs_human
    }

def human_handoff(state: TicketState):
    """人工转接节点"""
    return {
        "status": "escalated",
        "answer": f"您的问题已转接人工客服，工单号: {state['ticket_id']}。请稍候，将有专人为您服务。"
    }

def auto_resolve(state: TicketState):
    """自动解决节点"""
    return {"status": "resolved"}

def should_escalate(state: TicketState):
    """决定是否需要人工处理"""
    if state["needs_human"]:
        return "human_handoff"
    else:
        return "auto_resolve"

# 构建工作流图
workflow = StateGraph(TicketState)

# 添加节点
workflow.add_node("classify", classify_question)
workflow.add_node("search_kb", search_knowledge_base)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("human_handoff", human_handoff)
workflow.add_node("auto_resolve", auto_resolve)

# 添加边
workflow.set_entry_point("classify")
workflow.add_edge("classify", "search_kb")
workflow.add_edge("search_kb", "generate_answer")
workflow.add_conditional_edges(
    "generate_answer",
    should_escalate,
    {
        "human_handoff": "human_handoff",
        "auto_resolve": "auto_resolve"
    }
)
workflow.add_edge("human_handoff", END)
workflow.add_edge("auto_resolve", END)

# 编译应用
checkpointer = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=checkpointer)

# 使用示例
def handle_customer_inquiry(question: str, customer_id: str):
    initial_state = {
        "question": question,
        "category": "",
        "priority": "low",
        "ticket_id": "",
        "knowledge_result": "",
        "answer": "",
        "needs_human": False,
        "status": "open"
    }
    
    config = {"configurable": {"thread_id": f"customer-{customer_id}"}}
    result = app.invoke(initial_state, config=config)
    
    return {
        "ticket_id": result["ticket_id"],
        "answer": result["answer"],
        "status": result["status"],
        "category": result["category"]
    }

# 测试用例
test_questions = [
    "我的账单有问题，费用不对",
    "网站无法登录，一直显示错误",
    "如何更改个人信息？"
]

for i, question in enumerate(test_questions):
    print(f"\n客户 {i+1}: {question}")
    result = handle_customer_inquiry(question, str(i+1))
    print(f"工单号: {result['ticket_id']}")
    print(f"状态: {result['status']}")
    print(f"回答: {result['answer']}")
```

**LangGraph 优势：**
- 完整的工单生命周期管理
- 智能路由和人工转接
- 持久化状态跟踪
- 可扩展的工作流设计

---

## 案例3：多智能体研究助手

### 需求描述
构建一个研究助手系统，包含：
1. 研究员智能体：负责信息收集
2. 分析师智能体：负责数据分析
3. 编辑智能体：负责报告生成
4. 监督者智能体：协调各智能体工作

### LangChain 实现的困难

使用 LangChain 实现多智能体系统会遇到以下挑战：

```python
# LangChain 的多智能体实现会很复杂且不够优雅
from langchain.agents import AgentExecutor
from langchain.tools import Tool

# 需要手动管理智能体间的通信
def researcher_agent():
    # 研究员逻辑
    pass

def analyst_agent():
    # 分析师逻辑  
    pass

def editor_agent():
    # 编辑逻辑
    pass

# 需要复杂的协调逻辑
def coordinate_agents():
    # 手动协调各智能体
    research_result = researcher_agent()
    analysis_result = analyst_agent(research_result)
    final_report = editor_agent(analysis_result)
    return final_report
```

**问题：**
- 缺乏原生的多智能体支持
- 状态在智能体间传递困难
- 无法实现复杂的协作模式

### LangGraph 实现（多智能体系统）

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from typing import TypedDict, Literal

class ResearchState(TypedDict):
    task: str
    research_data: str
    analysis_result: str
    final_report: str
    current_agent: str
    iteration_count: int
    max_iterations: int

# 创建工具
search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun()

def create_researcher_agent():
    """创建研究员智能体"""
    llm = ChatOpenAI(model="gpt-4")
    tools = [search_tool, wiki_tool]
    
    system_prompt = """你是一个专业的研究员。
    你的任务是收集和整理相关信息。
    请使用可用的工具进行深入研究。"""
    
    return create_react_agent(llm, tools, system_prompt)

def create_analyst_agent():
    """创建分析师智能体"""
    llm = ChatOpenAI(model="gpt-4")
    
    system_prompt = """你是一个数据分析师。
    你的任务是分析研究数据，找出关键洞察和趋势。
    请提供结构化的分析结果。"""
    
    return create_react_agent(llm, [], system_prompt)

def create_editor_agent():
    """创建编辑智能体"""
    llm = ChatOpenAI(model="gpt-4")
    
    system_prompt = """你是一个专业编辑。
    你的任务是将分析结果整理成清晰、专业的报告。
    请确保报告结构清晰、语言准确。"""
    
    return create_react_agent(llm, [], system_prompt)

# 智能体节点
def researcher_node(state: ResearchState):
    """研究员节点"""
    researcher = create_researcher_agent()
    
    result = researcher.invoke({
        "messages": [("human", f"请研究以下主题: {state['task']}")]
    })
    
    return {
        "research_data": result["messages"][-1].content,
        "current_agent": "analyst"
    }

def analyst_node(state: ResearchState):
    """分析师节点"""
    analyst = create_analyst_agent()
    
    prompt = f"""
    基于以下研究数据进行分析:
    {state['research_data']}
    
    请提供详细的分析和洞察。
    """
    
    result = analyst.invoke({
        "messages": [("human", prompt)]
    })
    
    return {
        "analysis_result": result["messages"][-1].content,
        "current_agent": "editor"
    }

def editor_node(state: ResearchState):
    """编辑节点"""
    editor = create_editor_agent()
    
    prompt = f"""
    基于以下研究数据和分析结果，生成最终报告:
    
    研究数据: {state['research_data']}
    分析结果: {state['analysis_result']}
    
    请生成一份专业的研究报告。
    """
    
    result = editor.invoke({
        "messages": [("human", prompt)]
    })
    
    return {
        "final_report": result["messages"][-1].content,
        "current_agent": "complete"
    }

def supervisor_node(state: ResearchState):
    """监督者节点 - 决定下一步行动"""
    if state["current_agent"] == "researcher":
        return "researcher"
    elif state["current_agent"] == "analyst":
        return "analyst"
    elif state["current_agent"] == "editor":
        return "editor"
    else:
        return END

# 构建多智能体工作流
workflow = StateGraph(ResearchState)

# 添加智能体节点
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("editor", editor_node)

# 设置路由
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "editor")
workflow.add_edge("editor", END)

# 编译应用
research_app = workflow.compile()

# 使用示例
def conduct_research(topic: str):
    initial_state = {
        "task": topic,
        "research_data": "",
        "analysis_result": "",
        "final_report": "",
        "current_agent": "researcher",
        "iteration_count": 0,
        "max_iterations": 3
    }
    
    result = research_app.invoke(initial_state)
    
    return {
        "topic": topic,
        "research_data": result["research_data"],
        "analysis": result["analysis_result"],
        "report": result["final_report"]
    }

# 测试研究任务
research_topic = "人工智能在医疗诊断中的应用现状和发展趋势"
result = conduct_research(research_topic)

print(f"研究主题: {result['topic']}")
print(f"\n最终报告:\n{result['report']}")
```

**LangGraph 多智能体优势：**
- 原生多智能体支持
- 清晰的智能体间通信机制
- 灵活的工作流编排
- 状态在智能体间自然流转

---

## 案例4：代码审查自动化

### 需求描述
构建一个自动化代码审查系统：
1. 分析代码质量
2. 检查安全漏洞
3. 生成改进建议
4. 需要人工最终确认

### LangGraph 实现（Human-in-the-Loop）

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, List

class CodeReviewState(TypedDict):
    code: str
    language: str
    quality_score: float
    security_issues: List[str]
    suggestions: List[str]
    human_approved: bool
    final_decision: str

def analyze_code_quality(state: CodeReviewState):
    """代码质量分析节点"""
    llm = ChatOpenAI()
    
    prompt = f"""
    分析以下 {state['language']} 代码的质量，给出1-10的评分：
    
    ```{state['language']}
    {state['code']}
    ```
    
    请考虑：代码可读性、性能、最佳实践、错误处理等因素。
    """
    
    response = llm.invoke(prompt)
    
    # 简化评分提取
    quality_score = 7.5  # 实际应该从响应中解析
    
    return {"quality_score": quality_score}

def check_security(state: CodeReviewState):
    """安全检查节点"""
    llm = ChatOpenAI()
    
    prompt = f"""
    检查以下代码的安全问题：
    
    ```{state['language']}
    {state['code']}
    ```
    
    请列出可能的安全漏洞和风险。
    """
    
    response = llm.invoke(prompt)
    
    # 模拟安全问题检测
    security_issues = [
        "潜在的SQL注入风险",
        "未验证用户输入"
    ] if "input" in state['code'] else []
    
    return {"security_issues": security_issues}

def generate_suggestions(state: CodeReviewState):
    """生成改进建议节点"""
    llm = ChatOpenAI()
    
    prompt = f"""
    基于代码质量评分 {state['quality_score']} 和安全问题 {state['security_issues']}，
    为以下代码生成改进建议：
    
    ```{state['language']}
    {state['code']}
    ```
    """
    
    response = llm.invoke(prompt)
    
    suggestions = [
        "添加输入验证",
        "改进错误处理",
        "优化性能"
    ]
    
    return {"suggestions": suggestions}

def human_review_required(state: CodeReviewState):
    """判断是否需要人工审核"""
    return (
        state['quality_score'] < 6.0 or 
        len(state['security_issues']) > 0
    )

def human_review_node(state: CodeReviewState):
    """人工审核节点"""
    # 在实际应用中，这里会暂停执行等待人工输入
    print(f"代码质量评分: {state['quality_score']}")
    print(f"安全问题: {state['security_issues']}")
    print(f"改进建议: {state['suggestions']}")
    
    # 模拟人工决策
    human_input = input("是否批准此代码? (y/n): ")
    approved = human_input.lower() == 'y'
    
    decision = "批准" if approved else "需要修改"
    
    return {
        "human_approved": approved,
        "final_decision": decision
    }

def auto_approve_node(state: CodeReviewState):
    """自动批准节点"""
    return {
        "human_approved": True,
        "final_decision": "自动批准"
    }

def route_decision(state: CodeReviewState):
    """路由决策"""
    if human_review_required(state):
        return "human_review"
    else:
        return "auto_approve"

# 构建代码审查工作流
workflow = StateGraph(CodeReviewState)

# 添加节点
workflow.add_node("quality_check", analyze_code_quality)
workflow.add_node("security_check", check_security)
workflow.add_node("generate_suggestions", generate_suggestions)
workflow.add_node("human_review", human_review_node)
workflow.add_node("auto_approve", auto_approve_node)

# 添加边
workflow.set_entry_point("quality_check")
workflow.add_edge("quality_check", "security_check")
workflow.add_edge("security_check", "generate_suggestions")
workflow.add_conditional_edges(
    "generate_suggestions",
    route_decision,
    {
        "human_review": "human_review",
        "auto_approve": "auto_approve"
    }
)
workflow.add_edge("human_review", END)
workflow.add_edge("auto_approve", END)

# 配置检查点以支持人工干预
checkpointer = SqliteSaver.from_conn_string(":memory:")
code_review_app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"]  # 在人工审核前暂停
)

# 使用示例
def review_code(code: str, language: str = "python"):
    initial_state = {
        "code": code,
        "language": language,
        "quality_score": 0.0,
        "security_issues": [],
        "suggestions": [],
        "human_approved": False,
        "final_decision": ""
    }
    
    config = {"configurable": {"thread_id": f"review-{hash(code)}"}}
    
    # 第一阶段：自动分析
    result = code_review_app.invoke(initial_state, config=config)
    
    # 如果需要人工审核，继续执行
    if not result.get("final_decision"):
        result = code_review_app.invoke(None, config=config)
    
    return result

# 测试代码审查
test_code = """
def process_user_data(user_input):
    # 处理用户数据
    result = eval(user_input)  # 安全风险！
    return result
"""

review_result = review_code(test_code)
print(f"\n最终决策: {review_result['final_decision']}")
```

---

## 总结

通过以上实践案例可以看出：

### LangChain 适合：
- **简单对话系统**: 快速实现，代码简洁
- **线性工作流**: 数据处理管道，文档处理
- **原型验证**: 快速测试想法和概念

### LangGraph 适合：
- **复杂业务流程**: 工单系统，审批流程
- **多智能体系统**: 协作型 AI 应用
- **人机协作场景**: 需要人工干预和决策
- **状态管理**: 长期会话，跨会话持久化

### 选择建议：
1. **从简单开始**: 先用 LangChain 验证核心功能
2. **按需升级**: 当需要复杂控制流时迁移到 LangGraph
3. **混合使用**: 在同一项目中根据模块复杂度选择不同框架