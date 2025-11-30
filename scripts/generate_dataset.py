# scripts/generate_dataset.py
import random
import csv
from pathlib import Path

spam_templates = [
    "Congratulations! You won {amount} cash prize. Claim now.",
    "URGENT! Your ATM card will be blocked. Verify at {link}.",
    "Free recharge of {amount} available. Click {link}.",
    "You have been selected for a {item}. Reply YES to claim.",
    "Your loan of {amount} is pre-approved. Apply now.",
    "Winner! You are chosen for a lucky draw. Call {number}.",
    "Exclusive offer: Buy 1 get 1 free. Limited time.",
    "Your account will be suspended. Update KYC here: {link}.",
    "Get a free gift voucher worth {amount}. Click {link}.",
    "You’ve won a holiday trip to {place}! Claim now."
]

ham_templates = [
    "Bro where are you?",
    "Call me when you reach.",
    "I will message you later.",
    "Are you coming today?",
    "Good morning, have a nice day!",
    "Don't forget the class at {time}.",
    "I reached home.",
    "Thank you!",
    "Ok done.",
    "See you tomorrow.",
    "Send me the notes.",
    "I'm waiting outside.",
    "Did you eat?",
    "On the way.",
    "I will call you back."
]

amounts = ["₹5000", "₹10000", "₹300", "$500", "₹2000"]
links = ["http://verify-now.com", "http://update-kyc.in", "http://claim-gift.com"]
items = ["free Netflix subscription", "special cashback offer", "VIP membership"]
numbers = ["9876543210", "9028374651", "9988776655"]
places = ["Goa", "Dubai", "Singapore"]
times = ["2 PM", "4 PM", "6 PM"]

data = []

# Generate 500 spam messages
for _ in range(500):
    template = random.choice(spam_templates)
    msg = template.format(
        amount=random.choice(amounts),
        link=random.choice(links),
        item=random.choice(items),
        number=random.choice(numbers),
        place=random.choice(places)
    )
    data.append(("spam", msg))

# Generate 500 ham messages
for _ in range(500):
    template = random.choice(ham_templates)
    msg = template.format(
        time=random.choice(times)
    )
    data.append(("ham", msg))

# Shuffle dataset
random.shuffle(data)

# Output path
output_path = Path("data/spam.csv")
output_path.parent.mkdir(exist_ok=True)

# Write to CSV
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "text"])
    writer.writerows(data)

print("Dataset created successfully!")
print("Path:", output_path.absolute())
print("Total rows:", len(data))
