# NOTES

The model only listens to audio before it stops. It uses three things to figure this out:
* how the speakers voice sounds in the last couple of seconds before the pause like how loud it is and if the pitch goes up or down
* what sound the speaker ends with like if they make a sharp stop or a soft sound
* what has been happening in the conversation so far like how long it has been since the last pause and how many pauses there have been

The model is trained to pay attention to how long the pauses are because only long pauses can cause problems.
It uses a combination of nine models to make a decision because one model was not good enough for both English and Hindi.
With this the model still has trouble with pauses where the speaker is thinking and then adds more to what they were saying.
It also has trouble, with short pauses, where it is hard to figure out what is going on.
English is harder for the model to understand than Hindi because in English pauses can sound like the end of a sentence.
