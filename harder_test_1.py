import os, glob, shutil
import storage, lancedb_store
storage._DB_PATH = "/tmp/hard.db"
lancedb_store.DB_PATH = "/tmp/hard_lance"
for f in glob.glob("/tmp/hard.db*"):
    os.remove(f)
shutil.rmtree("/tmp/hard_lance", ignore_errors=True)

from storage import get_db, process_fact
from neo_ollama import extract_facts

F = "F"   # statements: should produce facts
X = "X"   # filler: should produce no facts
TURNS = [
    (F, "I was born in Bengaluru."),
    (F, "I finished my schooling in Bengaluru."),
    (F, "I got my first job in Bengaluru."),
    (F, "I work at a startup in Bengaluru."),
    (F, "I am vegetarian."),
    (F, "I drink black coffee every morning."),
    (F, "I don't like cilantro."),
    (F, "I like playing chess."),
    (F, "I enjoy trail running on weekends."),
    (F, "I play the guitar."),
    (F, "I like to listen to jazz."),
    (F, "I stopped playing chess."),
    (F, "I moved to Germany for work."),
    (F, "I speak English and Kannada."),
    (F, "I am learning German."),
    (F, "I speak Hindi."),
    (F, "I studied robotics engineering."),
    (F, "I know Python well."),
    (F, "I am learning Rust."),
    (F, "I use Linux on my laptop."),
    (F, "I started eating meat again."),
    (X, "hi there"),
    (X, "thanks!"),
    (X, "explain what a decorator is in Python"),
    (X, "what is 17 times 23?"),
    (X, "write a haiku about rain"),
    (X, "how does a binary search work?"),
    (X, "ok cool"),
    (X, "can you summarize what a transformer is?"),
    (X, "what is the boiling point of water?"),
    (X, "good morning"),
    (X, "tell me a joke"),
    (X, "what does git rebase do?"),
    (X, "how many days are in a leap year?"),
    (X, "thank you, that helps"),
    (X, "what is a hash map?"),
    (F, "I moved from Germany to Liechtenstein."),
    (F, "My sister is younger than me."),
    (F, "My mother lives in Bengaluru."),
    (F, "My father is retired."),
    (F, "I have a dog named Bruno."),
    (F, "Forget that I have a dog named Bruno."),
    (F, "I prefer tea in the evening."),
    (F, "I like mountains more than beaches."),
    (F, "I have visited Japan."),
    (F, "I have visited Iceland."),
    (F, "I want to work in machine learning in Europe."),
    (F, "I want to learn to fly a drone."),
    (F, "I run a small blog about robotics."),
    (F, "I am saving money for a master's degree."),
    (F, "I am learning French."),
    (F, "I moved from Liechistan to France."),
    (F, "I own a bicycle."),
    (F, "I sold my bicycle."),
    (F, "I read science fiction."),
    (F, "My favourite author is Ted Chiang."),
    (F, "I celebrate Diwali with my family."),
    (F, "I am left-handed."),
    (F, "I play badminton on Fridays."),
    (F, "I prefer window seats on flights."),
    (F, "I volunteer at a coding club."),
    (F, "I own a mechanical keyboard."),
    (F, "I studied robotics engineering."),   # exact repeat, should be NONE
    (F, "I now work in Zurich."),
    (F, "I want to hike the Alps."),
]

filler_leaks = 0
for i, (kind, text) in enumerate(TURNS, 1):
    facts = extract_facts(text)
    if kind == X and facts:
        filler_leaks += 1
        print(f"{i} FILLER LEAK | {text} -> {facts}")
    with get_db() as conn:
        for fact in facts:
            result = process_fact(conn, fact)
            print(f"{i} | {fact} -> {result}")
    if kind == F and not facts:
        print(f"{i} | NO FACT EXTRACTED | {text}")

with get_db() as conn:
    n_active = conn.execute("SELECT COUNT(*) FROM facts WHERE status='active'").fetchone()[0]
    n_retracted = conn.execute("SELECT COUNT(*) FROM facts WHERE status='retracted'").fetchone()[0]
print("turns:", len(TURNS), "| active:", n_active, "| retracted:", n_retracted, "| filler leaks:", filler_leaks)