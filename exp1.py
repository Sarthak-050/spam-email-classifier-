import pandas as pd
import re

# ---------------- LOAD ----------------
df1 = pd.read_csv("combined_data.csv")
df2 = pd.read_csv("email_text.csv")
df3 = pd.read_csv("email_origin.csv")

# ---------------- RENAME ----------------
df1 = df1.rename(columns={'text': 'Message', 'label': 'Category'})
df2 = df2.rename(columns={'text': 'Message', 'label': 'Category'})
df3 = df3.rename(columns={'origin': 'Message', 'label': 'Category'})

# ---------------- MERGE ----------------
df = pd.concat([df1, df2, df3], ignore_index=True)

# ---------------- FIX LABELS ----------------
def fix_label(x):
    x = str(x).strip().lower()

    if x in ['spam', '1']:
        return 1
    elif x in ['ham', '0']:
        return 0
    else:
        return None
        

# Apply label fixing
df['label'] = df['Category'].apply(fix_label)

# ---------------- REMOVE INVALID ----------------

df = df.dropna(subset=['Message', 'label'])

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# Apply cleaning
df['cleaned'] = df['Message'].apply(clean_text)

# ---------------- FINAL CHECK ----------------
print("Final Shape:", df.shape)
print(df['label'].value_counts())
print(df[['Message', 'cleaned', 'label']].head())

# ---------------- SAVE ----------------
df.to_csv("final_dataset.csv", index=False)
