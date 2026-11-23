"""Tool-calling and code-writing prompts in the platform's formats, which the edit was NOT fitted on.

Purpose: the public suite contains BFCL and HumanEval/MBPP tasks, so the test may too. This
probe checks on which side of the task feature such prompts fall. The correct behaviour of the
edit on every one of them is "gain 1.0". All prompts are written here; none is a benchmark item.

- tool families: {"messages", "tools"} objects, rendered like delta_cli/public_eval/bfcl_handler.py
  does (tool schemas passed to the chat template). `tool_single`: one function, one call.
  `tool_multiple`: several functions, one applies. `tool_parallel`: several calls.
  `tool_irrelevant`: no listed function applies. `tool_followup`: a tool result is already
  in the conversation.
- code families: the two prompt wrappers of delta_cli/public_eval/coding.py (`code_complete`
  for HumanEval-style function completion, `code_task` for MBPP-style tasks with asserts).

usage: python make_tool_probe.py OUT.jsonl
"""
import hashlib
import json
import sys


def fn(name, description, **params):
    """A function schema; params are name=(type, description) or name=(type, description, False) if optional."""
    props = {k: {"type": v[0], "description": v[1]} for k, v in params.items()}
    for k, v in params.items():
        if v[0] == "array":
            props[k]["items"] = {"type": "string"}
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": [k for k, v in params.items() if len(v) < 3]}}


WEATHER = fn("get_weather", "Get the current weather for a city.", city=("string", "City name"), unit=("string", "celsius or fahrenheit", False))
CONVERT = fn("convert_currency", "Convert an amount of money between two currencies.", amount=("number", "Amount to convert"),
             source=("string", "ISO code of the source currency"), target=("string", "ISO code of the target currency"))
AREA = fn("triangle_area", "Area of a triangle from its base and height.", base=("number", "Base length"), height=("number", "Height"))
SEARCH = fn("search_flights", "Search flights between two airports on a date.", origin=("string", "IATA code of the origin"),
            destination=("string", "IATA code of the destination"), date=("string", "Date as YYYY-MM-DD"))
BOOK = fn("book_hotel", "Book a hotel room.", city=("string", "City"), nights=("integer", "Number of nights"), guests=("integer", "Number of guests"))
STOCK = fn("get_stock_price", "Latest price of a stock.", symbol=("string", "Ticker symbol"))
MAIL = fn("send_email", "Send an email.", to=("string", "Recipient address"), subject=("string", "Subject line"), body=("string", "Message text"))
REMIND = fn("create_reminder", "Create a reminder at a given time.", text=("string", "What to be reminded of"), time=("string", "Time as HH:MM, 24-hour"))
DIST = fn("distance_between", "Road distance in kilometres between two cities.", origin=("string", "Start city"), destination=("string", "End city"))
TRANSLATE = fn("translate_text", "Translate text into a target language.", text=("string", "Text to translate"), language=("string", "Target language"))
SORT = fn("sort_numbers", "Sort a list of numbers.", numbers=("array", "Numbers to sort"), order=("string", "ascending or descending", False))
FILES = fn("list_files", "List the files of a directory.", path=("string", "Directory path"))
READ = fn("read_file", "Read a text file.", path=("string", "File path"))

user = lambda text: [{"role": "user", "content": text}]
TOOLS = {
    "tool_single": [
        (user("What's the weather like in Lisbon right now, in celsius?"), [WEATHER]),
        (user("I have 250 US dollars. How many euros is that?"), [CONVERT]),
        (user("Find the area of a triangle with a base of 12 and a height of 7.5."), [AREA]),
        (user("Remind me to call the dentist at half past three in the afternoon."), [REMIND]),
    ],
    "tool_multiple": [
        (user("How far is it by road from Munich to Vienna?"), [WEATHER, DIST, SEARCH, STOCK]),
        (user("What is Tesla's stock trading at?"), [CONVERT, STOCK, WEATHER, MAIL]),
        (user("Translate 'where is the train station' into Japanese."), [TRANSLATE, MAIL, SEARCH, AREA]),
        (user("Look for flights from SFO to JFK on 2027-03-14."), [BOOK, SEARCH, DIST, WEATHER]),
    ],
    "tool_parallel": [
        (user("Compare the weather in Oslo and in Cairo for me."), [WEATHER, DIST]),
        (user("Get me the latest prices of AAPL, MSFT and NVDA."), [STOCK, CONVERT]),
        (user("Book a hotel in Rome for 3 nights for 2 guests, and find flights from LHR to FCO on 2027-05-02."), [BOOK, SEARCH, WEATHER]),
        (user("Convert 100 GBP to JPY and 100 GBP to CHF."), [CONVERT, STOCK]),
    ],
    "tool_irrelevant": [
        (user("Who wrote the novel 'Middlemarch'?"), [WEATHER, CONVERT]),
        (user("What is the derivative of x^3 + 2x?"), [SEARCH, BOOK, MAIL]),
        (user("Can you recommend a good name for a golden retriever puppy?"), [STOCK, DIST]),
        (user("Explain in two sentences why the sky is blue."), [SORT, TRANSLATE]),
    ],
    "tool_followup": [
        ([{"role": "user", "content": "Show me what is in /srv/reports and then open the newest quarterly summary."},
          {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "list_files", "arguments": {"path": "/srv/reports"}}}]},
          {"role": "tool", "content": json.dumps({"files": ["q1-summary.txt", "q2-summary.txt", "notes.txt"]})}], [FILES, READ]),
        ([{"role": "user", "content": "Is it warmer in Madrid or in Athens today? Email the answer to ana@example.com."},
          {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Madrid", "unit": "celsius"}}},
                                                              {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Athens", "unit": "celsius"}}}]},
          {"role": "tool", "content": json.dumps({"city": "Madrid", "temperature": 31})},
          {"role": "tool", "content": json.dumps({"city": "Athens", "temperature": 28})}], [WEATHER, MAIL]),
    ],
}

COMPLETE = "Complete this Python function. Return the complete function in a Python code block.\n```python\n{}\n```"
TASK = "Write Python code solving this task. Return code in a Python code block.\n{}\nThe code must pass:\n{}"
CODE = {
    "code_complete": [COMPLETE.format(src) for src in (
        'def running_max(numbers: list[int]) -> list[int]:\n    """ Return a list where element i is the largest value among numbers[0..i].\n    >>> running_max([1, 3, 2, 5, 4])\n    [1, 3, 3, 5, 5]\n    """\n',
        'def count_vowels(text: str) -> int:\n    """ Count the vowels (a, e, i, o, u, upper or lower case) in text.\n    >>> count_vowels("Hello World")\n    3\n    """\n',
        'from typing import List\n\n\ndef merge_sorted(a: List[int], b: List[int]) -> List[int]:\n    """ Merge two ascending lists into one ascending list.\n    >>> merge_sorted([1, 4, 9], [2, 3, 10])\n    [1, 2, 3, 4, 9, 10]\n    """\n',
        'def is_balanced(brackets: str) -> bool:\n    """ brackets is a string of "(", ")", "[" and "]".\n    Return True if every bracket is closed in the right order.\n    >>> is_balanced("([])")\n    True\n    >>> is_balanced("([)]")\n    False\n    """\n',
        'def digit_sum(n: int) -> int:\n    """ Return the sum of the decimal digits of the absolute value of n.\n    >>> digit_sum(-482)\n    14\n    """\n',
        'def rotate_left(items: list, k: int) -> list:\n    """ Rotate items to the left by k positions (k may exceed the length).\n    >>> rotate_left([1, 2, 3, 4, 5], 2)\n    [3, 4, 5, 1, 2]\n    """\n',
    )],
    "code_task": [TASK.format(text, "\n".join(tests)) for text, tests in (
        ("Write a function to find the second smallest distinct number in a list.",
         ["assert second_smallest([4, 1, 3, 1, 2]) == 2", "assert second_smallest([10, 20]) == 20", "assert second_smallest([5, 5, 7]) == 7"]),
        ("Write a function to check whether a given string is a palindrome, ignoring spaces and letter case.",
         ['assert is_palindrome("Never odd or even") == True', 'assert is_palindrome("delta") == False', 'assert is_palindrome("A") == True']),
        ("Write a function that returns the number of words in a sentence that are longer than n characters.",
         ['assert count_long_words("the quick brown fox", 3) == 2', 'assert count_long_words("a bb ccc", 5) == 0', 'assert count_long_words("stream of tokens", 2) == 2']),
        ("Write a function to compute the greatest common divisor of all numbers in a list.",
         ["assert gcd_list([12, 18, 24]) == 6", "assert gcd_list([7, 13]) == 1", "assert gcd_list([40]) == 40"]),
        ("Write a function to remove consecutive duplicate characters from a string.",
         ['assert squeeze("aaabccdd") == "abcd"', 'assert squeeze("abc") == "abc"', 'assert squeeze("") == ""']),
        ("Write a function to return the indices of the two numbers in a list that add up to a target, as a tuple in ascending order.",
         ["assert pair_with_sum([2, 7, 11, 15], 9) == (0, 1)", "assert pair_with_sum([3, 2, 4], 6) == (1, 2)", "assert pair_with_sum([1, 5, 5], 10) == (1, 2)"]),
    )],
}

jobs = []
for family, items in TOOLS.items():
    for messages, tools in items:
        prompt = {"messages": messages, "tools": tools}
        key = hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()[:10]
        jobs.append({"id": f"{family}:{key}", "family": family, "prompt": prompt, "scorer": "none"})
for family, prompts in CODE.items():
    for prompt in prompts:
        jobs.append({"id": f"{family}:" + hashlib.sha256(prompt.encode()).hexdigest()[:10], "family": family, "prompt": prompt, "scorer": "none"})
with open(sys.argv[1], "w") as f:
    for job in jobs:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")
print(f"{len(jobs)} prompts in {len(TOOLS) + len(CODE)} formats -> {sys.argv[1]}")
