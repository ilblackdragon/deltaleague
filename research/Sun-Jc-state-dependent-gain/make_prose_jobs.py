"""Write a set of requests whose answers are text to be delivered, not reasoning.

Used to collect model states for fitting the "answer writing" feature, so that the feature is
not fitted on evaluation items. Everything is generated here from small lists; nothing is taken
from a benchmark. Four families:
  prose   a piece of writing (kind of text x topic x optional format requirement, 11 languages)
  short   a question to be answered with a word, a number or a short phrase and nothing else
  struct  a JSON object, a list, a table row or a function call in a requested layout
  code    a small function or command

usage: python make_prose_jobs.py OUT.jsonl [--n 160] [--seed 7]   (--n counts prose; the other families are fixed lists)
"""
import hashlib
import json
import random
import sys

KINDS = [
    "Write a short story about {t}.",
    "Write a product description for {t}.",
    "Write an email to a colleague about {t}.",
    "Write the opening of a blog post about {t}.",
    "Write a poem about {t}.",
    "Write a short speech about {t}.",
    "Write a dialogue between two friends about {t}.",
    "Explain {t} to a ten-year-old.",
    "Write a review of {t}.",
    "Write a letter of complaint about {t}.",
    "Write a step-by-step guide to {t}.",
    "Write an advertisement for {t}.",
    "Write a news article about {t}.",
    "Give five practical tips about {t}.",
    "Write a diary entry about {t}.",
    "Summarize the main arguments for and against {t}.",
    "Write a thank-you note about {t}.",
    "Describe {t} for a travel brochure.",
    "Write a job posting related to {t}.",
    "Write a short essay on {t}.",
]
TOPICS = [
    "a lighthouse keeper's last night on duty", "a reusable water bottle", "moving the weekly meeting to Thursday",
    "learning to bake bread", "the first snow of the year", "volunteering in your town", "a lost umbrella",
    "how rainbows form", "a neighbourhood noodle restaurant", "a delayed train", "planting a small herb garden",
    "a second-hand bicycle shop", "a new bridge opening in a small city", "keeping a desk tidy", "a long hike in the hills",
    "remote work", "a birthday present from a grandparent", "a quiet fishing village", "a public library",
    "the history of the pencil", "why the sea is salty", "a robot that learns to paint", "a noisy upstairs neighbour",
    "saving money on groceries", "a school science fair", "electric buses", "a cat that follows the postman",
    "how vaccines train the immune system", "a mountain hut in winter", "an old family recipe",
    "choosing a first musical instrument", "a street market at dawn", "recycling glass", "a chess club for beginners",
    "the night sky far from city lights", "a broken washing machine", "starting to run in the mornings",
    "a museum of maps", "tides and the moon", "a bakery that opens at 4 a.m.",
]
FORMATS = [
    "", "", "", "",
    "Use exactly three paragraphs.",
    "Do not use any commas in your answer.",
    "Write the whole answer in lowercase letters.",
    "Write the whole answer in capital letters.",
    "Start your answer with a title wrapped in double angle brackets, like <<title>>.",
    "End your answer with a postscript that begins with P.S.",
    "Wrap your entire answer in double quotation marks.",
    "Use at least eight sentences.",
    "Keep it under sixty words.",
    "Use exactly four bullet points in markdown.",
    "Highlight at least two phrases with markdown, like *this*.",
    "Finish with the exact sentence: Is there anything else I can help with?",
    "Do not use the words 'very' or 'really'.",
    "Use the word 'window' at least three times.",
    "Answer in French.", "Answer in Spanish.", "Answer in German.", "Answer in Portuguese.", "Answer in Italian.",
    "Answer in Chinese.", "Answer in Japanese.", "Answer in Korean.", "Answer in Russian.", "Answer in Hindi.", "Answer in Arabic.",
    "Give two different versions separated by a line with six asterisks.",
    "Format the answer as a JSON object with the keys \"title\" and \"text\".",
    "First repeat this request word for word, then write your answer.",
]

SHORT = [
    "What is the capital of Australia? Reply with the city name only.",
    "Is 91 a prime number? Answer with exactly one word: yes or no.",
    "Which is heavier, a kilogram of feathers or a kilogram of iron? Answer in at most five words.",
    "What is the past tense of the verb 'to seek'? Reply with one word.",
    "How many days are there in a leap year? Reply with the number only.",
    "Which planet is known as the Red Planet? One word.",
    "What is the chemical symbol for sodium? Reply with the symbol only.",
    "Translate the word 'library' into Spanish. Reply with the single word.",
    "What colour do you get when you mix blue and yellow paint? One word.",
    "Sort these words alphabetically and reply with the sorted list only, separated by commas: pear, apple, mango, cherry.",
    "What is the opposite of 'scarce'? Reply with one word.",
    "Which month comes right before October? Reply with the month only.",
    "How many sides does a hexagon have? Reply with the number only.",
    "Is a tomato botanically a fruit? Answer yes or no and nothing else.",
    "Who wrote the play 'Hamlet'? Reply with the name only.",
    "What is the plural of 'mouse' (the animal)? One word.",
    "Which is the largest ocean on Earth? Reply with its name only.",
    "Write the number 47 in Roman numerals. Reply with the numeral only.",
    "What is the freezing point of water in degrees Fahrenheit? Reply with the number only.",
    "Which word does not belong: oak, pine, tulip, birch? Reply with that word only.",
    "Spell the word 'rhythm' backwards. Reply with the letters only.",
    "What is the next letter in the sequence B, D, F, H? Reply with one letter.",
    "Is the statement 'All squares are rectangles' true or false? One word.",
    "What is the smallest prime number greater than 20? Reply with the number only.",
    "Name the longest river in Africa. Reply with the name only.",
    "How many minutes are there in two and a half hours? Reply with the number only.",
    "What is the comparative form of the adjective 'good'? One word.",
    "Which gas do plants absorb from the air for photosynthesis? Reply with its name only.",
    "Unscramble the letters 'tca' into an English word for an animal. Reply with the word only.",
    "In which direction does the sun rise? One word.",
    "What is 15 percent of 200? Reply with the number only.",
    "Which of these is a mammal: shark, dolphin, trout? Reply with that word only.",
    "Give a synonym of 'quick' that starts with the letter r. One word.",
    "How many continents are there? Reply with the number only.",
    "What is the first name of the scientist Einstein? Reply with the name only.",
    "Does the word 'banana' contain the letter 'e'? Answer yes or no only.",
    "What do you call a baby dog? One word.",
    "Which is larger, 3/4 or 5/8? Reply with the fraction only.",
    "What is the square root of 144? Reply with the number only.",
    "Complete the proverb with one word: 'Practice makes ...'",
    "Classify the sentiment of this review as positive or negative, one word: 'The soup was cold and the waiter ignored us.'",
    "Classify the sentiment of this review as positive or negative, one word: 'Lovely room, friendly staff, we will come back.'",
    "Which word in this sentence is the verb? 'The old dog slept soundly.' Reply with that word only.",
    "What day comes two days after Friday? Reply with the day only.",
    "What is the boiling point of water at sea level in degrees Celsius? Reply with the number only.",
    "Which of the numbers 12, 19, 27 is odd and divisible by 3? Reply with the number only.",
    "Is Lisbon the capital of Spain? Answer yes or no only.",
    "What is the abbreviation of 'kilometre'? Reply with the abbreviation only.",
]
STRUCT = [
    "You can call the function set_alarm(time, label). The user says: wake me at 6:45 for my flight. Reply with only the function call.",
    "You can call the function send_message(to, text). The user says: tell Maria that the meeting moved to 3 pm. Reply with only the function call.",
    "You can call the function convert(amount, from_unit, to_unit). The user says: how many miles is 12 kilometres? Reply with only the function call.",
    "You can call the function add_event(title, date, start, end). The user says: put a dentist appointment on 14 March from 9 to 10. Reply with only the function call.",
    "You can call the function search_flights(origin, destination, date). The user says: find me flights from Oslo to Rome on 2 June. Reply with only the function call.",
    "You can call the function play(song, artist). The user says: play Yesterday by The Beatles. Reply with only the function call.",
    "You can call the function set_thermostat(room, celsius). The user says: make the bedroom 19 degrees. Reply with only the function call.",
    "You can call the function translate(text, target_language). The user says: how do you say good morning in Japanese? Reply with only the function call.",
    "You can call the functions get_weather(city) and get_time(city). The user says: what time is it in Nairobi? Reply with only the function call.",
    "You can call the functions get_weather(city) and get_time(city). The user says: do I need an umbrella in Bergen today? Reply with only the function call.",
    "You can call the function create_reminder(text, when). The user says: remind me to water the plants tomorrow morning. Reply with only the function call.",
    "You can call the function order_pizza(size, toppings). The user says: a large pizza with mushrooms and olives please. Reply with only the function call.",
    "You can call the function book_table(restaurant, people, time). The user says: a table for four at Luigi's at 8 pm. Reply with only the function call.",
    "You can call the function calculate_tip(bill, percent). The user says: what is an 18 percent tip on 54 dollars? Reply with only the function call.",
    "You can call the function lookup_word(word, language). The user says: what does the German word 'Fernweh' mean? Reply with only the function call.",
    "You can call the function start_timer(minutes). The user says: time my eggs for seven minutes. Reply with only the function call.",
    "Return a JSON object with the keys \"name\", \"age\" and \"city\" for this sentence: 'Tom is 31 and lives in Leeds.' Output only the JSON.",
    "Return a JSON object with the keys \"product\", \"price\" and \"currency\" for this sentence: 'The lamp costs 45 euros.' Output only the JSON.",
    "Return a JSON array of the colours mentioned in this sentence: 'She wore a red coat, green boots and a yellow scarf.' Output only the JSON.",
    "Return a JSON object mapping each of these words to its length: tree, river, cloud. Output only the JSON.",
    "Extract the date and the place from 'The fair opens in Lyon on 5 May' as a JSON object with keys \"date\" and \"place\". Output only the JSON.",
    "Write one CSV line with the header fields name, country and year for the invention 'telephone, United States, 1876'. Output only the header line and the data line.",
    "List the first five multiples of 7 as a comma-separated list and nothing else.",
    "List the days of the weekend separated by a semicolon and nothing else.",
    "Write the ingredients of a basic pancake batter as a numbered list with one ingredient per line and nothing else.",
    "Give the three primary colours as a markdown table with one column named Colour. Output only the table.",
    "Write a YAML snippet with the keys host, port and debug set to localhost, 8080 and false. Output only the YAML.",
    "Write an HTML unordered list with the items tea, coffee and juice. Output only the HTML.",
    "Fill in the template with the given values and output only the result. Template: 'Dear {name}, your order {id} ships on {day}.' Values: name=Ana, id=A-17, day=Tuesday.",
    "Rewrite the date '3 February 2021' in the format YYYY-MM-DD. Output only the date.",
    "Convert the time '4:30 pm' to 24-hour format. Output only the time.",
    "Write the phone number 5551234567 in the format (555) 123-4567. Output only the number.",
    "Give the key-value pairs colour=blue, size=medium and price=20 as one line separated by vertical bars. Output only that line.",
    "Write a regular expression that matches a five-digit number. Output only the expression.",
    "Write the SQL query that selects the name and age of every row in the table users where age is above 30. Output only the query.",
    "Write the shell command that lists all files ending in .txt in the current directory. Output only the command.",
]
CODE = [
    "Write a Python function is_even(n) that returns True when n is even. Output only the code.",
    "Write a Python function reverse_words(s) that reverses the order of the words in a sentence. Output only the code.",
    "Write a Python function count_vowels(s) that counts the vowels in a string. Output only the code.",
    "Write a Python function factorial(n) using a loop. Output only the code.",
    "Write a Python function that returns the largest of three numbers. Output only the code.",
    "Write a Python function fizzbuzz(n) that returns a list of the FizzBuzz values from 1 to n. Output only the code.",
    "Write a Python function is_palindrome(s) that ignores case and spaces. Output only the code.",
    "Write a Python function mean(values) that returns the average of a non-empty list. Output only the code.",
    "Write a Python function celsius_to_fahrenheit(c). Output only the code.",
    "Write a Python function unique(items) that removes duplicates but keeps the order. Output only the code.",
    "Write a Python function merge_dicts(a, b) where values from b win. Output only the code.",
    "Write a Python function word_lengths(s) that maps each word to its length. Output only the code.",
    "Write a Python function second_largest(values). Output only the code.",
    "Write a Python function that checks whether a year is a leap year. Output only the code.",
    "Write a Python function gcd(a, b) using the Euclidean algorithm. Output only the code.",
    "Write a Python function flatten(lists) that flattens a list of lists. Output only the code.",
    "Write a JavaScript function that returns the sum of an array of numbers. Output only the code.",
    "Write a JavaScript function that capitalises the first letter of a string. Output only the code.",
    "Write a Python function chunk(items, size) that splits a list into pieces of the given size. Output only the code.",
    "Write a Python function that returns the n-th Fibonacci number iteratively. Output only the code.",
    "Write a Python function count_words(text) that returns a dictionary of word counts. Output only the code.",
    "Write a Python function clamp(x, low, high). Output only the code.",
    "Write a Python function that returns True if two strings are anagrams. Output only the code.",
    "Write a Python function to_snake_case(name) that converts camelCase to snake_case. Output only the code.",
]

out = sys.argv[1]
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 160
rng = random.Random(int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 7)
seen, jobs = set(), []
while len(jobs) < n:
    prompt = rng.choice(KINDS).format(t=rng.choice(TOPICS))
    fmt = rng.choice(FORMATS)
    if fmt:
        prompt += " " + fmt
        if rng.random() < 0.25:
            extra = rng.choice([f for f in FORMATS if f and f != fmt and not (f.startswith("Answer in") and fmt.startswith("Answer in"))])
            prompt += " " + extra
    if prompt in seen:
        continue
    seen.add(prompt)
    jobs.append({"id": "prose:" + hashlib.sha256(prompt.encode()).hexdigest()[:10], "family": "prose", "prompt": prompt, "scorer": "none"})
for family, prompts in (("short", SHORT), ("struct", STRUCT), ("code", CODE)):
    for prompt in prompts:
        jobs.append({"id": f"{family}:" + hashlib.sha256(prompt.encode()).hexdigest()[:10], "family": family, "prompt": prompt, "scorer": "none"})
with open(out, "w") as f:
    for job in jobs:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")
print(f"{len(jobs)} prompts -> {out}")
