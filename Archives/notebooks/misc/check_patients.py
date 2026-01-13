"""Check and seed patients in database."""
import sqlite3

conn = sqlite3.connect('drug_interactions.db')
c = conn.cursor()

# Check tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print('Tables:', tables)

# Check if diabetic_patients exists
if 'diabetic_patients' in tables:
    c.execute('SELECT COUNT(*) FROM diabetic_patients')
    count = c.fetchone()[0]
    print(f'Diabetic patients: {count}')
    
    if count > 0:
        c.execute('SELECT patient_id, name FROM diabetic_patients LIMIT 10')
        for row in c.fetchall():
            print(f'  - {row[0]}: {row[1]}')
else:
    print('Table diabetic_patients does NOT exist!')

conn.close()
