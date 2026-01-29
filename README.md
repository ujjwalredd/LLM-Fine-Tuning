# WhatsApp Chat Style Fine Tuning

## Why I Built This

Look I'll be honest with you. One day I was scrolling through my WhatsApp chats with my friend and I thought what if I could train an AI to sound exactly like me when texting. Not because I'm lazy or anything but purely for the science of it. Also it's pretty cool to see if machine learning can actually capture someone's texting personality.

I had thousands of messages just sitting there. Years of conversations with all my weird texting habits like saying ha all the time or using guru randomly or just being super casual with grammar. That's basically a goldmine of training data right? So I figured why not see if a language model could learn to be me.

The idea was simple. Take all those chat messages where my friend says something and I respond. Feed that to a language model. Train it. Then see if it can generate responses that sound like they actually came from me. Spoiler alert it worked better than I expected.

## What This Thing Actually Does

This project takes your WhatsApp chat export and turns it into a personalized AI that texts like you. The whole pipeline goes like this. First you export your chat history from WhatsApp. Then you run it through a preprocessing script that cleans up all the junk and creates conversation pairs. After that you fine tune a small language model on those pairs. Finally you end up with a model that can read a message and generate a response in your style.

Think of it like teaching a really smart parrot to talk except instead of just repeating phrases it actually understands context and tries to respond the way you would.

## The Technical Journey Or How I Learned to Stop Worrying and Love TinyLlama

### Attempt One The Disaster

I started super ambitious. I wanted to use Phi 3.5 mini instruct because hey bigger model means better results right? Wrong. I fired up Google Colab with its free T4 GPU and the thing immediately told me to get lost. Out of memory errors everywhere. Fine I thought I'll just use Unsloth to make training faster. That lasted about ten minutes before I hit some bizarre torchvision compatibility issue that made no sense.

At this point I was already an hour in and hadn't trained anything. Classic machine learning experience.

### Attempt Two Still Too Optimistic

Okay new plan. Forget Unsloth. I'll just use regular HuggingFace transformers with Phi 3.5 mini but I'll use quantization to compress the model. Surely that will fit in memory. I installed bitsandbytes and configured everything perfectly. Ran the training script. Immediately got quantization errors that I still don't fully understand.

I was starting to think maybe I should just learn to text better manually instead of training an AI to do it for me.

### Attempt Three The One That Worked

I gave up on being fancy and switched to TinyLlama 1.1B Chat. It's smaller. It's simpler. And guess what? It actually fit in the GPU memory and trained without exploding. Revolutionary concept I know.

The moral of the story is sometimes the boring solution is the right solution. Who would have thought.

## How the Data Processing Works

WhatsApp exports are messy. Like really messy. You get all these system messages about people joining groups or encryption notices or message deletions. None of that is useful for teaching an AI how you text.

So I built a parser that does the dirty work. It reads through the entire chat export line by line. Filters out all the system garbage that clutters everything. Groups together messages when you send multiple texts in a row because let's face it nobody sends just one message when three will do. Then it pairs up the conversations so you have your friend says X and you respond with Y.

Those pairs become the training data. Each pair is basically one example of your texting style in action.

The script also does some analysis to give you stats. How long are your messages on average? Do you use a lot of emojis? What phrases do you repeat constantly? It's kind of fun to see your texting habits quantified like that. Turns out I say ha way more than I realized.

After running my 5121 line chat export through this whole process I ended up with 1169 clean training pairs. Not huge by AI dataset standards but definitely enough to capture how I communicate with friends.

## The Training Setup

### Picking the Model

I went with TinyLlama 1.1B Chat v1.0. It's a small conversational model based on the Llama architecture but specifically trained for chatting. Perfect for learning texting styles without requiring a supercomputer to run.

Instead of fine tuning every single parameter in the model which would take forever and use insane amounts of memory I used something called LoRA. That stands for Low Rank Adaptation. Basically it freezes most of the model weights and only trains small adapter layers on top. Way more efficient and it works surprisingly well for style transfer.

### The Hyperparameters

LoRA rank was set to 64. That controls how many new parameters get trained. Learning rate was 2e-4 which is pretty standard for fine tuning tasks. Batch size was 4 because that's what fit in the T4 GPU without causing out of memory crashes.

I targeted the attention layers and MLP layers for training since those are where the model learns language patterns and conversational style. The whole setup was designed to be as memory efficient as possible while still being effective.

### First Training Run

Started conservative with 3 epochs. Didn't want to overdo it right out of the gate. The training took about 3 minutes total which was way faster than expected. Loss dropped from 1.53 to 1.01 which showed clear learning happening.

Got some warnings about gradient checkpointing and cache settings but nothing that actually broke anything. Sometimes you just get lucky and things work.

### Extended Training Run

After seeing good initial results I decided to keep going. Ended up training for 13 epochs total. The loss kept dropping smoothly all the way down to 0.045 which honestly looks great on paper.

But here's the thing. With only 1169 training examples doing 13 epochs means the model saw each conversation about 13 times. That's a lot of repetition. It might have learned my style really well. Or it might have just memorized my exact messages. Hard to say without testing.

## What I Learned Along the Way

### Library Conflicts Are Real

The number of times I hit compatibility issues between different Python libraries was honestly ridiculous. Torch version conflicts with transformers. Transformers conflicts with accelerate. Everything conflicts with bitsandbytes. You spend half your time debugging dependencies and the other half actually doing machine learning.

The lesson here is start simple. Don't try to use every fancy optimization technique at once. Get something basic working first then add complexity.

### Small Models Are Underrated

I was so focused on using the biggest model I could fit that I wasted time trying to make Phi 3.5 work. TinyLlama with 1.1 billion parameters ended up being perfect for this task. It's fast to train. Fits easily in free GPU memory. And it's plenty capable for learning conversational style.

You don't always need the latest and greatest. Sometimes a smaller focused model trained well beats a bigger model trained poorly.

### Data Quality Matters More Than Quantity

I only had 1169 training pairs but they were clean high quality examples of actual conversations. That turned out to be way more valuable than having thousands of messy poorly formatted examples. The preprocessing step where you filter junk and create proper conversation pairs makes a huge difference in results.

Garbage in garbage out is still true even with fancy AI models.

### Overfitting Is Sneaky

Looking at that loss curve dropping from 0.63 to 0.045 feels amazing. But with a small dataset and many epochs there's a real risk the model just memorized everything instead of learning general patterns. You need to test on new conversations to know if it actually generalized.

This is why validation sets exist. I should have split off 20 percent of the data for validation instead of using everything for training. Live and learn.

## How to Use This Project

### Step One Get Your Chat Data

Open WhatsApp on your phone. Go to the chat you want to use. Tap the three dots menu. Select More then Export Chat. Choose without media because we just need the text. This gives you a txt file with your entire conversation history.

Transfer that file to your computer wherever you're running the code.

### Step Two Preprocess the Data

Run the preprocessing script on your chat export. It will filter out system messages and create conversation pairs. You'll get a dataset file ready for training plus some stats about your texting patterns.

Check those stats to make sure the data looks reasonable. If your average message length is like 2 words something probably went wrong.

### Step Three Train the Model

Fire up Google Colab or whatever GPU environment you're using. Install the required libraries which are transformers peft datasets accelerate and trl. Load the TinyLlama model and configure LoRA training.

Point it at your preprocessed dataset and start training. With the settings I used it should take under 10 minutes for a few thousand message pairs.

Watch the loss go down. Feel smart. Wonder if you're overfitting. Train anyway.

### Step Four Test It Out

This is the fun part. Load your trained model and give it some test messages that weren't in the training data. See what it generates. Does it sound like you? Does it use your phrases and style? Or does it just output nonsense?

If it works congratulations you've successfully cloned your texting personality into an AI. If it doesn't work well at least you learned something about machine learning.

## What's Next

The obvious next step is to actually test this thing properly. I need to generate a bunch of responses and see if they genuinely sound like me or if the model just memorized my training data. Ideally I'd show the outputs to my friend without telling him which ones are real and which are AI generated and see if he can tell the difference.

I should also go back and add a validation split so I can track overfitting properly. And save checkpoints during training so I can roll back if later epochs make things worse.

Long term it would be cool to build a simple chat interface where you can have a conversation with AI you. Like texting yourself from the future or something. Weird but interesting.

## Project Structure
```
whatsapp-style-finetune/
├── data/
│   ├── sample_chat.txt
│   └── preprocessed_data.json
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── inference.py
├── models/
│   └── tinyllama-finetuned/
├── requirements.txt
└── README.md
```

## Requirements
```
transformers
peft
datasets
accelerate
trl
torch
```

## Final Thoughts

This was a fun experiment that actually worked way better than I expected. The fact that you can capture someone's texting style with just a thousand or so examples and a small language model is pretty wild. Machine learning has come a long way.

The hardest parts weren't even the AI stuff. They were dealing with library conflicts and figuring out what would fit in GPU memory. Once I settled on a simple setup with TinyLlama and LoRA everything came together quickly.

If you want to try this yourself just remember to start simple. Don't overthink it. Use a small model. Get something working first. Then optimize later if you need to.

And most importantly don't train for 13 epochs on a tiny dataset unless you want to risk memorization. But hey I had to learn that the hard way so now you don't have to.