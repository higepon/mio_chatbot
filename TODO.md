# Goal
Make *fun* chatbot like human.
# TODO
- [done] Understand how CS20SI chatbot works
  - 6/11 read her documents
  - 6/11 train
  - 6/11 chat
  - 6/11 draw its model structure
    - didn't draw, but here is pretty similar one
  - 6/11 Post it to my blog
  - 6/11 Read improvement section of her document
  - 6/11 Make improvement plan here
- Make basic chatbot
  - 6/11 Prepare pycharm env with python 3.3 and latest TensorFlow
  - 6/12 Port data.py to Python3
    - 6/12 fix O(1) slowness
    - 6/12 Compare generated file with the original ones.
  - 6/12 Investigate why data.py is so slow
  - Write own train.py
  - Very naive implmentation with the same data as CS20SI
  - Compare the two bots
  - Probably rewrite it in [new seq2seq API](https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention)
- Find a best way to record trial and error
  - model parameters (do we have to keep all the parameter files?)
  - How we construct model
  - Maybe tag or release?
  - Think what should be done while training? Read papers?
- Tune model
  - Make a list of tuning points
  - BEAM search
  - Re-read http://web.stanford.edu/class/cs20si/assignments/a3.pdf and make a list of todo again
- Improvement plan
- 1. Train on multiple datasets
- 2. Use more than just one utterance as the encoder
- 3. Make your chatbot remember information from the previous conversation
- 4. Create a chatbot with personality
- 5. Use character-level sequence to sequence model for the chatbot
- 6. Construct the response in a non-greedy way
- 7. Create a feedback loop that allows users to train your chatbot
- Make it work in English
- Make it work in Japanese
- Make tweet bot
- Make deploy process for tweet bot
- Make tweet bot available in cloud

# random ideas
- Can we use 2ch.net data?
