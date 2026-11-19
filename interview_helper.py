"""
AI Interview Helper Engine.
Analyzes transcribed questions from the interviewer (captured via system loopback)
and synthesizes real-time chatbot responses, code implementations, technical explanations,
and STAR-method responses tailored to the candidate's target job role and profile context.
Supports real-time token streaming for zero-latency response display.
"""

import os
import re
import time
from typing import Dict, Any, List, Optional, Callable


DEFAULT_ROLE = "Software Engineer / Technical Candidate"
DEFAULT_PROFILE_CONTEXT = (
    "Experienced professional with strong problem-solving skills, system architecture expertise, "
    "and practical hands-on delivery experience."
)

ANSWER_STYLES = [
    "Direct Spoken Response (Only the Answer)",
    "STAR Method (Situation, Task, Action, Result)",
    "Technical Deep-Dive / Code & Architecture",
    "Executive / Concise Summary",
]


# Extensive Offline Technical & Coding Knowledge Base
OFFLINE_KNOWLEDGE_BASE: Dict[str, str] = {
    "linear search": (
        "```python\n"
        "def linear_search(arr, target):\n"
        "    \"\"\"\n"
        "    Iterates through arr sequentially to find target.\n"
        "    Returns index of target if found, otherwise -1.\n"
        "    \"\"\"\n"
        "    for i in range(len(arr)):\n"
        "        if arr[i] == target:\n"
        "            return i\n"
        "    return -1\n"
        "\n"
        "# Example Usage:\n"
        "numbers = [10, 50, 30, 70, 80, 20]\n"
        "target = 70\n"
        "index = linear_search(numbers, target)\n"
        "print(f\"Element found at index: {index}\")  # Output: 3\n"
        "```\n\n"
        "**Complexity Analysis:**\n"
        "• Time Complexity: O(n) in worst/average case; O(1) in best case (target is at index 0).\n"
        "• Space Complexity: O(1) auxiliary space (in-place).\n"
        "• Advantage: Works on unsorted data without preprocessing overhead."
    ),
    "binary search": (
        "```python\n"
        "def binary_search(arr, target):\n"
        "    \"\"\"\n"
        "    Divide-and-conquer search on sorted array.\n"
        "    Returns index of target if found, otherwise -1.\n"
        "    \"\"\"\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left <= right:\n"
        "        mid = (left + right) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            left = mid + 1\n"
        "        else:\n"
        "            right = mid - 1\n"
        "    return -1\n"
        "\n"
        "# Example Usage:\n"
        "sorted_nums = [11, 22, 34, 45, 67, 89]\n"
        "print(binary_search(sorted_nums, 45))  # Output: 3\n"
        "```\n\n"
        "**Complexity Analysis:**\n"
        "• Time Complexity: O(log n) logarithmic time.\n"
        "• Space Complexity: O(1) iterative auxiliary space.\n"
        "• Prerequisite: The input array must be sorted."
    ),
    "bubble sort": (
        "```python\n"
        "def bubble_sort(arr):\n"
        "    n = len(arr)\n"
        "    for i in range(n):\n"
        "        swapped = False\n"
        "        for j in range(0, n - i - 1):\n"
        "            if arr[j] > arr[j + 1]:\n"
        "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
        "                swapped = True\n"
        "        if not swapped:\n"
        "            break\n"
        "    return arr\n"
        "\n"
        "# Example:\n"
        "print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))\n"
        "```\n\n"
        "**Complexity Analysis:**\n"
        "• Time Complexity: O(n²) worst/average case; O(n) best case (already sorted with swapped flag).\n"
        "• Space Complexity: O(1) in-place auxiliary space."
    ),
    "merge sort": (
        "```python\n"
        "def merge_sort(arr):\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    mid = len(arr) // 2\n"
        "    left = merge_sort(arr[:mid])\n"
        "    right = merge_sort(arr[mid:])\n"
        "    \n"
        "    merged, i, j = [], 0, 0\n"
        "    while i < len(left) and j < len(right):\n"
        "        if left[i] <= right[j]:\n"
        "            merged.append(left[i])\n"
        "            i += 1\n"
        "        else:\n"
        "            merged.append(right[j])\n"
        "            j += 1\n"
        "    merged.extend(left[i:])\n"
        "    merged.extend(right[j:])\n"
        "    return merged\n"
        "```\n\n"
        "**Complexity Analysis:**\n"
        "• Time Complexity: O(n log n) guaranteed in all cases.\n"
        "• Space Complexity: O(n) auxiliary memory. Stable sort."
    ),
    "quick sort": (
        "```python\n"
        "def quick_sort(arr):\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    pivot = arr[len(arr) // 2]\n"
        "    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n"
        "    right = [x for x in arr if x > pivot]\n"
        "    return quick_sort(left) + middle + quick_sort(right)\n"
        "```\n\n"
        "**Complexity Analysis:**\n"
        "• Time Complexity: O(n log n) average case; O(n²) worst case (poor pivot selection).\n"
        "• Space Complexity: O(log n) call stack space."
    ),
    "two sum": (
        "```python\n"
        "def two_sum(nums, target):\n"
        "    \"\"\"\n"
        "    Finds indices of two elements that add up to target.\n"
        "    Uses a hash map for single-pass O(n) lookup.\n"
        "    \"\"\"\n"
        "    seen = {}\n"
        "    for i, num in enumerate(nums):\n"
        "        diff = target - num\n"
        "        if diff in seen:\n"
        "            return [seen[diff], i]\n"
        "        seen[num] = i\n"
        "    return []\n"
        "\n"
        "# Example:\n"
        "print(two_sum([2, 7, 11, 15], 9))  # Output: [0, 1]\n"
        "```\n\n"
        "**Complexity Analysis:**\n"
        "• Time Complexity: O(n) single pass.\n"
        "• Space Complexity: O(n) hash table."
    ),
    "reverse linked list": (
        "```python\n"
        "class ListNode:\n"
        "    def __init__(self, val=0, next=None):\n"
        "        self.val = val\n"
        "        self.next = next\n"
        "\n"
        "def reverse_list(head):\n"
        "    prev = None\n"
        "    curr = head\n"
        "    while curr:\n"
        "        nxt = curr.next\n"
        "        curr.next = prev\n"
        "        prev = curr\n"
        "        curr = nxt\n"
        "    return prev\n"
        "```\n\n"
        "**Complexity Analysis:**\n"
        "• Time Complexity: O(n) visits each node once.\n"
        "• Space Complexity: O(1) in-place pointer manipulation."
    ),
    "valid parentheses": (
        "```python\n"
        "def is_valid_parentheses(s: str) -> bool:\n"
        "    bracket_map = {')': '(', '}': '{', ']': '['}\n"
        "    stack = []\n"
        "    for char in s:\n"
        "        if char in bracket_map:\n"
        "            top = stack.pop() if stack else '#'\n"
        "            if bracket_map[char] != top:\n"
        "                return False\n"
        "        else:\n"
        "            stack.append(char)\n"
        "    return len(stack) == 0\n"
        "```\n\n"
        "**Complexity Analysis:**\n"
        "• Time Complexity: O(n)\n"
        "• Space Complexity: O(n) stack memory."
    ),
    "fibonacci": (
        "```python\n"
        "def fibonacci(n):\n"
        "    if n <= 0: return 0\n"
        "    if n == 1: return 1\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
        "\n"
        "# Example:\n"
        "print([fibonacci(i) for i in range(8)])  # [0, 1, 1, 2, 3, 5, 8, 13]\n"
        "```\n\n"
        "**Complexity Analysis:**\n"
        "• Time Complexity: O(n) iterative dynamic programming.\n"
        "• Space Complexity: O(1) constant auxiliary space."
    ),
    "palindrome": (
        "```python\n"
        "def is_palindrome(s: str) -> bool:\n"
        "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
        "    left, right = 0, len(cleaned) - 1\n"
        "    while left < right:\n"
        "        if cleaned[left] != cleaned[right]:\n"
        "            return False\n"
        "        left += 1\n"
        "        right -= 1\n"
        "    return True\n"
        "```\n\n"
        "**Complexity Analysis:**\n"
        "• Time Complexity: O(n)\n"
        "• Space Complexity: O(n) cleaned string."
    ),
    "sql second highest salary": (
        "```sql\n"
        "-- Method 1: Using Subquery\n"
        "SELECT MAX(salary) AS SecondHighestSalary\n"
        "FROM Employee\n"
        "WHERE salary < (SELECT MAX(salary) FROM Employee);\n"
        "\n"
        "-- Method 2: Using LIMIT and OFFSET (MySQL/PostgreSQL)\n"
        "SELECT DISTINCT salary AS SecondHighestSalary\n"
        "FROM Employee\n"
        "ORDER BY salary DESC\n"
        "LIMIT 1 OFFSET 1;\n"
        "```"
    ),
    "tcp vs udp": (
        "• **TCP (Transmission Control Protocol):**\n"
        "  - Connection-oriented (Requires 3-way handshake: SYN, SYN-ACK, ACK).\n"
        "  - Reliability: Guarantees delivery, ordering, and error-checking via acknowledgments and retransmissions.\n"
        "  - Flow & Congestion Control: Uses sliding window protocols to prevent network congestion.\n"
        "  - Used for: HTTP/HTTPS (Web browsing), SSH, FTP, Email (SMTP).\n\n"
        "• **UDP (User Datagram Protocol):**\n"
        "  - Connectionless (Fire-and-forget; no handshake).\n"
        "  - Performance: Minimal overhead and zero latency delays; packets may arrive out-of-order or drop.\n"
        "  - Used for: Real-time Video Streaming, Voice over IP (VoIP), Online Gaming, DNS queries."
    ),
    "rest vs graphql": (
        "• **REST (Representational State Transfer):**\n"
        "  - Multiple URL endpoints for each resource (`/users`, `/users/1/orders`).\n"
        "  - Fixed data structures: Prone to over-fetching (retrieving unused data) or under-fetching (requiring multiple round-trip requests).\n"
        "  - Leverages HTTP status codes and standard HTTP caching (ETags, CDN).\n\n"
        "• **GraphQL:**\n"
        "  - Single POST endpoint (`/graphql`).\n"
        "  - Client-driven schema: Clients query exactly the fields they need in a single request.\n"
        "  - Solves over/under-fetching, making it ideal for mobile apps and complex microservices."
    ),
    "acid properties": (
        "• **Atomicity:** All operations within a transaction either succeed completely or roll back entirely (All-or-nothing).\n"
        "• **Consistency:** Ensures the database transitions only from one valid state to another, strictly enforcing all constraints, triggers, and foreign keys.\n"
        "• **Isolation:** Concurrent transactions execute without cross-contamination. Managed via isolation levels (Read Committed, Repeatable Read, Serializable).\n"
        "• **Durability:** Once a transaction commits, its changes survive permanently in non-volatile storage even after a system crash or power outage."
    ),
    "process vs thread": (
        "• **Process:**\n"
        "  - An independent executing instance of a program with isolated address space, heap, and system resources.\n"
        "  - High context switching overhead. Inter-Process Communication (IPC) requires sockets, pipes, or shared memory.\n\n"
        "• **Thread:**\n"
        "  - The smallest unit of CPU execution within a process. Threads in the same process share memory (code and heap) but have private stacks.\n"
        "  - Low context switching cost. Requires concurrency primitives (mutexes, locks) to prevent race conditions."
    ),
}


class InterviewHelperEngine:
    """Generates real-time interview answers tailored to the candidate's role and profile context."""

    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.conversation_history: List[Dict[str, str]] = []

    def set_config(self, provider: str, api_key: Optional[str], model_name: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name

    def clear_history(self):
        self.conversation_history.clear()

    def generate_interview_response_stream(
        self,
        question: str,
        role: str = DEFAULT_ROLE,
        profile_context: str = DEFAULT_PROFILE_CONTEXT,
        style: str = "Direct Spoken Response (Only the Answer)",
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Stream the direct response token-by-token for zero-latency chatbot behavior.
        Calls on_token(chunk_text) as tokens arrive and returns the full completed text.
        """
        if not question or not question.strip():
            msg = "Please provide an interview question."
            if on_token:
                on_token(msg)
            return msg

        # Resolve available LLM credentials
        active_provider, active_key = self._resolve_active_llm_provider()

        if active_provider in ("gemini", "google") and active_key:
            try:
                return self._generate_gemini_stream(question, active_key, role, profile_context, style, on_token)
            except Exception as e:
                print(f"[InterviewHelper] Gemini streaming error: {e}")

        elif active_provider == "groq" and active_key:
            try:
                return self._generate_groq_stream(question, active_key, role, profile_context, style, on_token)
            except Exception as e:
                print(f"[InterviewHelper] Groq streaming error: {e}")

        elif active_provider == "openai" and active_key:
            try:
                return self._generate_openai_stream(question, active_key, role, profile_context, style, on_token)
            except Exception as e:
                print(f"[InterviewHelper] OpenAI streaming error: {e}")

        # Intelligent offline fallback
        return self._generate_smart_fallback_stream(question, role, on_token)

    def generate_interview_response(
        self,
        question: str,
        role: str = DEFAULT_ROLE,
        profile_context: str = DEFAULT_PROFILE_CONTEXT,
        style: str = "Direct Spoken Response (Only the Answer)",
    ) -> Dict[str, Any]:
        """Synchronous wrapper for backwards compatibility."""
        full_text = self.generate_interview_response_stream(question, role, profile_context, style, on_token=None)
        return {"formatted_answer": full_text, "raw_text": full_text}

    def _resolve_active_llm_provider(self):
        """Finds the first available LLM key in config or environment."""
        if self.provider in ("gemini", "google") and (self.api_key or os.environ.get("GEMINI_API_KEY")):
            return "gemini", (self.api_key or os.environ.get("GEMINI_API_KEY"))
        if self.provider == "groq" and (self.api_key or os.environ.get("GROQ_API_KEY")):
            return "groq", (self.api_key or os.environ.get("GROQ_API_KEY"))
        if self.provider == "openai" and (self.api_key or os.environ.get("OPENAI_API_KEY")):
            return "openai", (self.api_key or os.environ.get("OPENAI_API_KEY"))

        # Automatic key fallback
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini", os.environ.get("GEMINI_API_KEY")
        if os.environ.get("GROQ_API_KEY"):
            return "groq", os.environ.get("GROQ_API_KEY")
        if os.environ.get("OPENAI_API_KEY"):
            return "openai", os.environ.get("OPENAI_API_KEY")

        return None, None

    def _build_system_prompt(self, role: str, profile_context: str, style: str) -> str:
        return (
            f"You are an elite, highly intelligent AI technical interview copilot and real-time coding assistant.\n"
            f"Candidate Role: {role}\n"
            f"Candidate Profile & Experience: {profile_context}\n"
            f"Tone & Style: {style}\n\n"
            "CRITICAL INSTRUCTIONS - ACT LIKE AN ADAPTIVE EXPERT CHATBOT:\n"
            "1. ADAPT EXACTLY TO THE QUESTION TYPE:\n"
            "   - If the user/interviewer asks for CODE (e.g. 'The code for linear search ?', 'write code for...', 'implement X'):\n"
            "     Directly provide the clean, working, idiomatic code inside a markdown code block (e.g. ```python ... ```), followed by a brief 1-2 sentence explanation and Time & Space Complexity. DO NOT write generic paragraphs when code is asked.\n"
            "   - If the user/interviewer asks a CONCEPTUAL or TECHNICAL QUESTION (e.g. 'What is X?', 'Difference between A and B'):\n"
            "     Provide a direct, concise, high-clarity technical explanation with key bullet points.\n"
            "   - If the user/interviewer asks a SYSTEM DESIGN or ARCHITECTURE question:\n"
            "     Provide high-level architecture, core components, and scalability considerations clearly.\n"
            "   - If the user/interviewer asks a BEHAVIORAL or SITUATIONAL question:\n"
            "     Provide a structured, first-person response using the STAR method.\n"
            "2. DO NOT output conversational filler like 'Sure! Here is the answer:' or 'Great question!'.\n"
            "3. DO NOT output forced template headings like 'Quick Pitch' or 'Traps to Avoid'.\n"
            "4. Output the direct, precise, high-quality answer immediately."
        )

    def _generate_gemini_stream(self, question: str, key: str, role: str, profile_context: str, style: str, on_token: Optional[Callable[[str], None]]) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key.strip())
        model = self.model_name or "gemini-flash-lite-latest"
        sys_prompt = self._build_system_prompt(role, profile_context, style)

        history_context = ""
        if self.conversation_history:
            recent = self.conversation_history[-3:]
            history_context = "\n\nRecent Interview Context:\n" + "\n".join(
                f"Interviewer: {h['q']}\nCandidate: {h['a']}" for h in recent
            )

        user_prompt = f"Interviewer Question: {question}{history_context}"

        full_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                temperature=0.2,
            )
        ):
            if chunk.text:
                full_text += chunk.text
                if on_token:
                    on_token(chunk.text)

        self.conversation_history.append({"q": question, "a": full_text})
        return full_text

    def _generate_groq_stream(self, question: str, key: str, role: str, profile_context: str, style: str, on_token: Optional[Callable[[str], None]]) -> str:
        from groq import Groq
        client = Groq(api_key=key.strip())
        model = "llama-3.3-70b-versatile"
        sys_prompt = self._build_system_prompt(role, profile_context, style)

        messages = [{"role": "system", "content": sys_prompt}]
        for h in self.conversation_history[-2:]:
            messages.append({"role": "user", "content": h["q"]})
            messages.append({"role": "assistant", "content": h["a"]})
        messages.append({"role": "user", "content": f"Interviewer Question: {question}"})

        chat = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            stream=True
        )

        full_text = ""
        for chunk in chat:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                if on_token:
                    on_token(delta)

        self.conversation_history.append({"q": question, "a": full_text})
        return full_text

    def _generate_openai_stream(self, question: str, key: str, role: str, profile_context: str, style: str, on_token: Optional[Callable[[str], None]]) -> str:
        import requests
        import json
        headers = {
            "Authorization": f"Bearer {key.strip()}",
            "Content-Type": "application/json"
        }
        sys_prompt = self._build_system_prompt(role, profile_context, style)
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Interviewer Question: {question}"}
            ],
            "temperature": 0.2,
            "stream": True
        }
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, stream=True, timeout=20)
        full_text = ""
        for line in res.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    try:
                        data = json.loads(line_str[6:])
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            full_text += delta
                            if on_token:
                                on_token(delta)
                    except Exception:
                        pass
        self.conversation_history.append({"q": question, "a": full_text})
        return full_text

    def _generate_smart_fallback_stream(self, question: str, role: str, on_token: Optional[Callable[[str], None]]) -> str:
        """
        Intelligent offline chatbot responder with rich data structures, algorithms,
        and technical interview concepts. Simulates fast token streaming.
        """
        clean_q = question.strip()
        ql = clean_q.lower()

        # 1. Check exact/fuzzy match in knowledge base
        matched_content = None
        for key, answer in OFFLINE_KNOWLEDGE_BASE.items():
            if key in ql:
                matched_content = answer
                break

        # 2. Check if asking for general code
        if not matched_content:
            is_code_request = any(w in ql for w in ["code", "function", "implement", "algorithm", "syntax", "program", "write"])
            if is_code_request:
                # Extract topic
                topic = re.sub(r'^(the\s+)?(code\s+for\s+|write\s+code\s+for\s+|how\s+to\s+write\s+|implement\s+)', '', ql, flags=re.IGNORECASE).strip(' ?.')
                func_name = re.sub(r'[^a-zA-Z0-9_]+', '_', topic).strip('_') or "solution"
                matched_content = (
                    f"```python\n"
                    f"def {func_name}(*args, **kwargs):\n"
                    f"    \"\"\"\n"
                    f"    Implementation for {topic.title()}.\n"
                    f"    \"\"\"\n"
                    f"    # Core logic\n"
                    f"    result = None\n"
                    f"    return result\n"
                    f"\n"
                    f"# Example Usage:\n"
                    f"# output = {func_name}()\n"
                    f"# print(output)\n"
                    f"```\n\n"
                    f"**Approach & Complexity:**\n"
                    f"• Approach: Clarify inputs/outputs, validate edge cases.\n"
                    f"• Time Complexity: O(n) average.\n"
                    f"• Space Complexity: O(1) auxiliary space."
                )

        # 3. Default contextual technical answer
        if not matched_content:
            topic = clean_q if len(clean_q) < 60 else "this problem"
            matched_content = (
                f"When addressing {topic} in production as a {role}, I focus on establishing clear requirements and identifying core trade-offs. "
                f"I break the problem into modular components with clean abstractions, comprehensive unit test coverage, and defensive error handling. "
                f"In my previous work, we prioritized observability, caching where appropriate, and gradual rollouts to maintain high reliability and performance."
            )

        # Stream words smoothly to the UI
        words = re.split(r'(\s+)', matched_content)
        full_text = ""
        for w in words:
            full_text += w
            if on_token:
                on_token(w)
            time.sleep(0.008)  # 8ms per token for fluid typewriter animation

        self.conversation_history.append({"q": question, "a": full_text})
        return full_text
