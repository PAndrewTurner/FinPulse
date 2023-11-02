import pandas as pd
import random
import string

def random_string(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

data = {
    'User_Identification': [random_string(6) for _ in range(2500)],
    'Date_of_Transaction': pd.date_range(start='2022-01-01', periods=2500, freq='D'),
    'Amount_of_Transaction': [round(random.uniform(10, 5000), 2) for _ in range(2500)],
    'Category': [random.choice(['Groceries', 'Entertainment', 'Utilities', 'Travel']) for _ in range(2500)]
}

df = pd.DataFrame(data)

file_path = r"C:\Users\jchin\Documents\Finpulse\sample_financial_data.csv"

# Save the data to the specified directory
df.to_csv(file_path, index=False)

print(f"Sample data for 2,500 rows saved to: {file_path}")
