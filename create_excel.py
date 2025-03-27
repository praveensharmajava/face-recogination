import pandas as pd
from datetime import datetime

# Sample data
data = {
    'Name': ['Praveen_photo', 'Praveer Sharma[Idiot]', 'image'],
    'Date_of_Birth': ['1990-01-01', '1995-02-15', '1988-12-31'],
    'LinkedIn_URL': [
        'https://linkedin.com/in/praveen-sharma',
        'https://linkedin.com/in/praveer-sharma',
        'https://linkedin.com/in/sample'
    ],
    'Phone_Number': ['+1234567890', '+9876543210', '+1122334455'],
    'Email_ID': ['praveen@example.com', 'praveer@example.com', 'sample@example.com']
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
excel_file = 'personal_info.xlsx'
df.to_excel(excel_file, index=False)
print(f"Excel file '{excel_file}' has been created successfully!") 