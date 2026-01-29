import re
import json
from datetime import datetime
from collections import defaultdict

def parse_whatsapp_chat(file_path, your_name="Ujjwal Reddy K S"):
    """
    Parse WhatsApp chat export and create fine-tuning dataset.
    """
    
    # Pattern for WhatsApp message format
    pattern = r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}:\d{2}\s*[AP]M)\]\s*([^:]+):\s*(.*)'
    
    messages = []
    system_indicators = [
        'end-to-end encrypted',
        'disappearing messages',
        'Tap to change',
        'Tap to update',
        'missed voice call',
        'missed video call',
        'deleted this message',
        'This message was deleted',
        '<Media omitted>',
        'image omitted',
        'video omitted',
        'audio omitted',
        'sticker omitted',
        'GIF omitted',
        'document omitted',
        'Contact card omitted',
        'location:',
        'Live location shared',
        '‎You',  # System messages often start with this
    ]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    for line in lines:
        match = re.match(pattern, line)
        if match:
            timestamp_str, sender, text = match.groups()
            sender = sender.strip()
            text = text.strip()
            
            # Skip system messages
            if any(indicator in text for indicator in system_indicators):
                continue
            if any(indicator in line for indicator in system_indicators):
                continue
            
            # Skip empty messages
            if not text or text in ['', '‎', ' ']:
                continue
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, "%m/%d/%y, %I:%M:%S %p")
            except:
                try:
                    timestamp = datetime.strptime(timestamp_str, "%d/%m/%y, %I:%M:%S %p")
                except:
                    continue
            
            messages.append({
                'timestamp': timestamp,
                'sender': sender,
                'text': text,
                'is_you': sender == your_name
            })
    
    print(f"Parsed {len(messages)} valid messages")
    return messages


def group_consecutive_messages(messages, time_gap_minutes=5):
    """
    Group consecutive messages from same sender into single turns.
    """
    if not messages:
        return []
    
    grouped = []
    current_group = {
        'sender': messages[0]['sender'],
        'texts': [messages[0]['text']],
        'is_you': messages[0]['is_you'],
        'timestamp': messages[0]['timestamp']
    }
    
    for msg in messages[1:]:
        time_diff = (msg['timestamp'] - current_group['timestamp']).total_seconds() / 60
        
        # Same sender and within time gap -> merge
        if msg['sender'] == current_group['sender'] and time_diff <= time_gap_minutes:
            current_group['texts'].append(msg['text'])
            current_group['timestamp'] = msg['timestamp']
        else:
            # Save current group and start new one
            grouped.append({
                'sender': current_group['sender'],
                'text': '\n'.join(current_group['texts']),
                'is_you': current_group['is_you']
            })
            current_group = {
                'sender': msg['sender'],
                'texts': [msg['text']],
                'is_you': msg['is_you'],
                'timestamp': msg['timestamp']
            }
    
    # Don't forget last group
    grouped.append({
        'sender': current_group['sender'],
        'text': '\n'.join(current_group['texts']),
        'is_you': current_group['is_you']
    })
    
    print(f"Grouped into {len(grouped)} conversation turns")
    return grouped


def create_training_pairs(grouped_messages, min_input_len=1, min_output_len=1):
    """
    Create (other person's message -> your reply) pairs for fine-tuning.
    """
    pairs = []
    
    for i in range(len(grouped_messages) - 1):
        current = grouped_messages[i]
        next_msg = grouped_messages[i + 1]
        
        # We want: other person speaks -> you reply
        if not current['is_you'] and next_msg['is_you']:
            input_text = current['text'].strip()
            output_text = next_msg['text'].strip()
            
            if len(input_text) >= min_input_len and len(output_text) >= min_output_len:
                pairs.append({
                    'input': input_text,
                    'output': output_text
                })
    
    print(f"Created {len(pairs)} training pairs")
    return pairs


def create_contextual_pairs(grouped_messages, context_turns=2):
    """
    Create pairs with conversation context (last N turns).
    Better for capturing conversation flow.
    """
    pairs = []
    
    for i in range(len(grouped_messages) - 1):
        current = grouped_messages[i]
        next_msg = grouped_messages[i + 1]
        
        if not current['is_you'] and next_msg['is_you']:
            # Build context from previous turns
            context_start = max(0, i - context_turns * 2)
            context = []
            
            for j in range(context_start, i):
                msg = grouped_messages[j]
                role = "You" if msg['is_you'] else msg['sender'].split()[0]
                context.append(f"{role}: {msg['text']}")
            
            # Add current message
            context.append(f"{current['sender'].split()[0]}: {current['text']}")
            
            pairs.append({
                'context': '\n'.join(context),
                'input': current['text'],
                'output': next_msg['text']
            })
    
    print(f"Created {len(pairs)} contextual training pairs")
    return pairs


def save_as_jsonl(pairs, output_path, format_type='alpaca'):
    """
    Save training pairs in different formats.
    
    Formats:
    - 'alpaca': Alpaca-style instruction format
    - 'chatml': ChatML format for most models
    - 'simple': Simple input/output pairs
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            if format_type == 'alpaca':
                record = {
                    "instruction": "Reply to this message in Ujjwal's texting style:",
                    "input": pair['input'],
                    "output": pair['output']
                }
            elif format_type == 'chatml':
                record = {
                    "messages": [
                        {"role": "system", "content": "You are Ujjwal. Reply in his casual texting style - short messages, informal language, uses 'ha', 'guru', emojis occasionally."},
                        {"role": "user", "content": pair['input']},
                        {"role": "assistant", "content": pair['output']}
                    ]
                }
            elif format_type == 'contextual':
                record = {
                    "messages": [
                        {"role": "system", "content": "You are Ujjwal. Reply in his casual texting style."},
                        {"role": "user", "content": f"Conversation:\n{pair.get('context', pair['input'])}\n\nReply as Ujjwal:"},
                        {"role": "assistant", "content": pair['output']}
                    ]
                }
            else:  # simple
                record = {
                    "input": pair['input'],
                    "output": pair['output']
                }
            
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Saved to {output_path} in {format_type} format")


def analyze_your_style(messages, your_name="Ujjwal Reddy K S"):
    """
    Analyze texting patterns for the style profile.
    """
    your_messages = [m['text'] for m in messages if m['is_you']]
    
    if not your_messages:
        return {}
    
    # Basic stats
    avg_length = sum(len(m) for m in your_messages) / len(your_messages)
    
    # Emoji usage
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0]')
    emoji_count = sum(len(emoji_pattern.findall(m)) for m in your_messages)
    
    # Common words/phrases
    all_words = ' '.join(your_messages).lower().split()
    word_freq = defaultdict(int)
    for word in all_words:
        word_freq[word] += 1
    
    top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:20]
    
    # Punctuation habits
    ends_with_period = sum(1 for m in your_messages if m.strip().endswith('.'))
    ends_with_emoji = sum(1 for m in your_messages if emoji_pattern.search(m[-2:] if len(m) > 1 else m))
    
    style = {
        'total_messages': len(your_messages),
        'avg_message_length': round(avg_length, 1),
        'emoji_per_message': round(emoji_count / len(your_messages), 2),
        'ends_with_period_pct': round(ends_with_period / len(your_messages) * 100, 1),
        'ends_with_emoji_pct': round(ends_with_emoji / len(your_messages) * 100, 1),
        'top_words': top_words,
        'sample_messages': your_messages[:10]
    }
    
    return style


# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    
    # Configuration
    CHAT_FILE = "/Users/ujjwalreddyks/Desktop/Desktop/FIne Tune LLM Poject/_chat.txt"  # Your WhatsApp export file
    YOUR_NAME = "Ujjwal Reddy K S"  # Your name as it appears in chat
    OUTPUT_FILE = "/Users/ujjwalreddyks/Desktop/Desktop/FIne Tune LLM Poject/training_data.jsonl"
    
    # Step 1: Parse
    messages = parse_whatsapp_chat(CHAT_FILE, YOUR_NAME)
    
    # Step 2: Analyze your style
    print("\n=== YOUR TEXTING STYLE ===")
    style = analyze_your_style(messages, YOUR_NAME)
    for k, v in style.items():
        if k != 'sample_messages' and k != 'top_words':
            print(f"{k}: {v}")
    print(f"Top words: {[w[0] for w in style.get('top_words', [])[:10]]}")
    print(f"\nSample messages:")
    for msg in style.get('sample_messages', [])[:5]:
        print(f"  - {msg}")
    
    # Step 3: Group consecutive messages
    grouped = group_consecutive_messages(messages, time_gap_minutes=5)
    
    # Step 4: Create training pairs
    pairs = create_training_pairs(grouped)
    
    # Step 5: Save in ChatML format (works with most fine-tuning tools)
    save_as_jsonl(pairs, OUTPUT_FILE, format_type='chatml')
    
    # Also save contextual version
    contextual_pairs = create_contextual_pairs(grouped, context_turns=2)
    save_as_jsonl(contextual_pairs, "training_data_contextual.jsonl", format_type='contextual')
    
    print(f"\n=== DONE ===")
    print(f"Training pairs: {len(pairs)}")
    print(f"Ready for fine-tuning!")