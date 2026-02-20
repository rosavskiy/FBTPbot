# -*- coding: utf-8 -*-
"""Полный анализ заявок ТП для классификации и подготовки к индексации."""
import pandas as pd
import numpy as np

path = r'd:\Projects\FBTPAI\real_support\tickets.xlsx'

print("Loading Excel (102K+ rows)...")
df = pd.read_excel(path, header=None, skiprows=2)
df.columns = [
    'Сотрудник', 'Организация', 'Тип_заявки', 
    'Текст_заявки', 'Комментарий_исполнителя', 'Отдел',
    'Статус', 'Не_принятые', 'В_очереди', 'В_процессе', 'Завершённые'
]

print(f"\nTotal rows: {len(df)}")

# --- Basic stats ---
print("\n=== COLUMN FILL RATES ===")
for col in df.columns:
    filled = df[col].notna().sum()
    nonempty = df[col].apply(lambda x: bool(str(x).strip()) if pd.notna(x) else False).sum()
    print(f"  {col}: filled={filled} non-empty={nonempty} ({nonempty/len(df)*100:.1f}%)")

# --- Тип заявки ---
print("\n=== ТИП ЗАЯВКИ (categories) ===")
types = df['Тип_заявки'].value_counts(dropna=False)
for t, cnt in types.items():
    print(f"  {t}: {cnt} ({cnt/len(df)*100:.1f}%)")

# --- Статус ---
print("\n=== СТАТУС ===")
statuses = df['Статус'].value_counts(dropna=False)
for s, cnt in statuses.items():
    print(f"  {s}: {cnt} ({cnt/len(df)*100:.1f}%)")

# --- Key: rows with both question AND answer ---
has_question = df['Текст_заявки'].notna() & (df['Текст_заявки'].astype(str).str.strip().str.len() > 5)
has_answer = df['Комментарий_исполнителя'].notna() & (df['Комментарий_исполнителя'].astype(str).str.strip().str.len() > 5)
has_both = has_question & has_answer

print(f"\n=== PAIR ANALYSIS ===")
print(f"  Has question (>5 chars): {has_question.sum()}")
print(f"  Has answer (>5 chars):   {has_answer.sum()}")
print(f"  Has BOTH:                {has_both.sum()}")
print(f"  Question only (no ans):  {(has_question & ~has_answer).sum()}")

# --- Pair quality analysis ---
pairs = df[has_both].copy()
pairs['q_len'] = pairs['Текст_заявки'].astype(str).str.strip().str.len()
pairs['a_len'] = pairs['Комментарий_исполнителя'].astype(str).str.strip().str.len()

print(f"\n=== PAIR QUALITY (n={len(pairs)}) ===")
print(f"  Question length: min={pairs['q_len'].min()}, median={pairs['q_len'].median():.0f}, max={pairs['q_len'].max()}")  
print(f"  Answer length:   min={pairs['a_len'].min()}, median={pairs['a_len'].median():.0f}, max={pairs['a_len'].max()}")

# Classify
# Gold: long answer (>=50), meaningful question (>=15)
# Silver: decent answer (>=30), some question (>=10)
# Bronze: short but exists
# Trash: too short, phone calls, "Принятый вызов"

trash_patterns = [
    'Принятый вызов', 'перезвонить', 'ПРИОРИТЕТ#', 
    'нет связи', 'Звонок', 'связаться', 'не дозвон'
]
trash_regex = '|'.join(trash_patterns)

pairs['is_trash_q'] = pairs['Текст_заявки'].astype(str).str.contains(trash_regex, case=False, na=False) | (pairs['q_len'] < 10)
pairs['is_trash_a'] = pairs['Комментарий_исполнителя'].astype(str).str.contains(trash_regex, case=False, na=False) | (pairs['a_len'] < 15)

# Assign quality
def classify(row):
    if row['is_trash_q'] or row['is_trash_a']:
        return 'trash'
    elif row['a_len'] >= 50 and row['q_len'] >= 15:
        return 'gold'
    elif row['a_len'] >= 30 and row['q_len'] >= 10:
        return 'silver'
    else:
        return 'bronze'

pairs['quality'] = pairs.apply(classify, axis=1)

print(f"\n=== QUALITY CLASSIFICATION ===")
for q in ['gold', 'silver', 'bronze', 'trash']:
    subset = pairs[pairs['quality'] == q]
    print(f"  {q.upper()}: {len(subset)} ({len(subset)/len(pairs)*100:.1f}%)")
    if len(subset) > 0:
        print(f"    Avg Q len: {subset['q_len'].mean():.0f}, Avg A len: {subset['a_len'].mean():.0f}")

# Type breakdown for pairs
print(f"\n=== PAIRS BY TYPE ===")
for q in ['gold', 'silver', 'bronze']:
    subset = pairs[pairs['quality'] == q]
    if len(subset) == 0:
        continue
    print(f"\n  --- {q.upper()} ({len(subset)}) ---")
    type_counts = subset['Тип_заявки'].value_counts()
    for t, cnt in type_counts.head(10).items():
        print(f"    {t}: {cnt}")

# Sample examples
for q in ['gold', 'silver', 'bronze', 'trash']:
    subset = pairs[pairs['quality'] == q]
    print(f"\n=== EXAMPLES: {q.upper()} (showing 3) ===")
    for _, row in subset.head(3).iterrows():
        question = str(row['Текст_заявки']).strip()[:150]
        answer = str(row['Комментарий_исполнителя']).strip()[:200]
        print(f"  Q: {question}")
        print(f"  A: {answer}")
        print(f"  Type: {row['Тип_заявки']} | Q:{row['q_len']} A:{row['a_len']}")
        print()

print("\nDone.")
