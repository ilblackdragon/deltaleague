"""A small set of reasoning prompts in formats that the edit was NOT fitted on.

Purpose: check on which side of the task feature unfamiliar reasoning formats fall (few-shot
question/answer prompts, step-by-step logic puzzles, ten-option multiple choice, true/false,
word problems in other languages). All prompts are written here; none is a benchmark item.
The correct behaviour of the edit on every one of them is "gain 1.0".

usage: python make_format_probe.py OUT.jsonl
"""
import hashlib
import json
import sys

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
P = {
    "fewshot": [
        "Question: A baker sells 14 loaves on Monday and twice as many on Tuesday. How many loaves does the baker sell in total?\nAnswer: On Tuesday the baker sells 14 * 2 = 28 loaves. In total 14 + 28 = 42 loaves. #### 42\n\nQuestion: Tom reads 12 pages a day for 5 days and then 20 pages on the sixth day. How many pages does he read?\nAnswer: In five days he reads 12 * 5 = 60 pages. With the sixth day 60 + 20 = 80 pages. #### 80\n\nQuestion: A farmer has 9 crates with 16 apples each and gives away 37 apples. How many apples are left?\nAnswer:",
        "Question: A train travels 60 km in the first hour and 45 km in each of the next two hours. How far does it travel?\nAnswer: 60 + 45 * 2 = 150 km. #### 150\n\nQuestion: A shop sells pencils at 3 for 2 dollars. How much do 18 pencils cost?\nAnswer: 18 / 3 = 6 groups, 6 * 2 = 12 dollars. #### 12\n\nQuestion: Maria has 5 boxes of 24 crayons and loses 17 crayons. She then shares the rest equally among 4 children. How many crayons does each child get, ignoring any remainder?\nAnswer:",
        "Problem: Simplify $3(x+4) - 2(x-1)$.\nSolution: $3x + 12 - 2x + 2 = x + 14$. Final Answer: The final answer is $x+14$. I hope it is correct.\n\nProblem: What is the sum of the first ten positive odd integers?\nSolution: The sum of the first $n$ odd integers is $n^2$, so the sum is $100$. Final Answer: The final answer is $100$. I hope it is correct.\n\nProblem: How many positive divisors does $72$ have?\nSolution:",
        "Problem: Solve for $x$: $2x + 7 = 31$.\nSolution: $2x = 24$, so $x = 12$. Final Answer: The final answer is $12$. I hope it is correct.\n\nProblem: A right triangle has legs of length 9 and 12. What is the length of its hypotenuse?\nSolution:",
    ],
    "logic": [
        "Q: Today is the last day of the first quarter of 2008. What is the date one week from today in MM/DD/YYYY?\nOptions:\n(A) 04/07/2008\n(B) 03/24/2008\n(C) 04/06/2008\n(D) 04/08/2008\nA: Let's think step by step.",
        "Q: Alice, Bob and Claire are playing a game. At the start Alice has a red ball, Bob has a blue ball and Claire has a green ball. First Alice and Bob swap balls. Then Bob and Claire swap balls. Finally Alice and Claire swap balls. At the end, Bob has the\nOptions:\n(A) red ball\n(B) blue ball\n(C) green ball\nA: Let's think step by step.",
        "Q: not ( True and False ) or ( False and not True ) is\nA: Let's think step by step.",
        "Q: The following statements each describe a set of five books on a shelf: a white book, a grey book, a red book, a green book and a black book. The grey book is to the left of the red book. The black book is the leftmost. The green book is the second from the right. The white book is to the right of the green book. Which book is in the middle?\nOptions:\n(A) white\n(B) grey\n(C) red\n(D) green\n(E) black\nA: Let's think step by step.",
    ],
    "mc10": [
        "Question: Which of the following organelles is the main site of ATP production in animal cells?\nOptions:\nA. Golgi apparatus\nB. Lysosome\nC. Mitochondrion\nD. Ribosome\nE. Nucleolus\nF. Peroxisome\nG. Centrosome\nH. Smooth endoplasmic reticulum\nI. Vacuole\nJ. Cell membrane\nAnswer: Let's think step by step.",
        "Question: A 2 kg object accelerates at 3 m/s^2 on a frictionless surface. What net force acts on it?\nOptions:\nA. 0.67 N\nB. 1.5 N\nC. 5 N\nD. 6 N\nE. 9 N\nF. 12 N\nG. 18 N\nH. 2 N\nI. 3 N\nJ. 60 N\nAnswer: Let's think step by step.",
        "Question: In economics, a good for which demand rises when consumer income falls is called\nOptions:\nA. a normal good\nB. a luxury good\nC. an inferior good\nD. a public good\nE. a complementary good\nF. a substitute good\nG. a capital good\nH. a Veblen good\nI. a merit good\nJ. a club good\nAnswer: Let's think step by step.",
        "Question: What is the time complexity of binary search on a sorted array of n elements?\nOptions:\nA. O(1)\nB. O(log n)\nC. O(n)\nD. O(n log n)\nE. O(n^2)\nF. O(sqrt n)\nG. O(2^n)\nH. O(n^3)\nI. O(log log n)\nJ. O(n!)\nAnswer: Let's think step by step.",
    ],
    "truefalse": [
        "Passage: The Amazon river discharges more water than any other river and flows mainly through Brazil.\nQuestion: does the amazon river flow through brazil?\nAnswer with yes or no." + SUFFIX,
        "Is the following statement true or false? The sum of two odd integers is always even." + SUFFIX,
        "Sentence: The trophy does not fit into the brown suitcase because it is too large. Does 'it' refer to the trophy? Answer yes or no." + SUFFIX,
        "Is 221 a prime number? Answer yes or no." + SUFFIX,
    ],
    "translated": [
        "一个水箱里有 240 升水。第一天用掉了三分之一，第二天用掉了剩下的四分之一。水箱里还剩多少升水？" + SUFFIX,
        "Un tren recorre 85 kilómetros por hora durante 3 horas y luego 60 kilómetros por hora durante 2 horas. ¿Cuántos kilómetros recorre en total?" + SUFFIX,
        "Ein Händler kauft 40 Hefte zu je 1,50 Euro und verkauft sie zu je 2,25 Euro. Wie viel Euro Gewinn macht er insgesamt?" + SUFFIX,
        "Dans une classe de 32 élèves, 5 élèves sur 8 font du sport. Parmi eux, la moitié joue au football. Combien d'élèves jouent au football ?" + SUFFIX,
        "Una ricetta richiede 3 uova ogni 250 grammi di farina. Quante uova servono per 1,5 chilogrammi di farina?" + SUFFIX,
        "Um reservatório enche a 15 litros por minuto e esvazia a 6 litros por minuto. Quantos minutos são necessários para acumular 180 litros?" + SUFFIX,
    ],
    "plainmath": [
        "What is 17 * 23 - 144 / 12?",
        "If f(x) = 2x^2 - 3x + 1, what is f(4)?",
        "A rectangle has a perimeter of 54 cm and its length is twice its width. What is its area in square centimetres?",
        "How many three-digit numbers are divisible by 7?",
    ],
}
jobs = []
for family, prompts in P.items():
    for prompt in prompts:
        jobs.append({"id": f"{family}:" + hashlib.sha256(prompt.encode()).hexdigest()[:10], "family": family, "prompt": prompt, "scorer": "none"})
with open(sys.argv[1], "w") as f:
    for job in jobs:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")
print(f"{len(jobs)} prompts in {len(P)} formats -> {sys.argv[1]}")
