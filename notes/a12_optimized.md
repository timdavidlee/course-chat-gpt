# Optimized Outputs

https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6081s


```python
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 
device = "cuda"
eval_iters = 200
n_emb = 384
n_head = 6
n_layer = 6
dropout = 0.2 
```

- larger batch size
- 256 characters of context to predict
- learning rate is reduced, because the NN is much larger
- many more heads , at 6 heads, its 384 / 6 = 64 dim as standards
- 6 layers
- 0.2 - every forward and backwards pass loses 20%

Running on a Nvidia 3090, sample output

```
python -m notes.b12b_train --gpu
================================================================================
using : cuda
================================================================================
================================================================================
model is on cuda!
================================================================================
  0%|                                                                                                   | 0/5000 [00:00<?, ?it/s]iter: 0 | train_loss: 4.4755    | valid_loss: 4.4713 cycle: 0:00:00.001323 total_time: 0:00:00.001325
 10%|████████▉                                                                                | 500/5000 [01:44<09:44,  7.70it/s]iter: 500       | train_loss: 2.0300    | valid_loss: 2.1061 cycle: 0:01:04.726900 total_time: 0:01:44.400129
 20%|█████████████████▌                                                                      | 1000/5000 [03:29<08:58,  7.42it/s]iter: 1,000     | train_loss: 1.6249    | valid_loss: 1.7993 cycle: 0:01:05.124363 total_time: 0:03:29.321880
 30%|██████████████████████████▍                                                             | 1500/5000 [05:14<07:31,  7.75it/s]iter: 1,500     | train_loss: 1.4493    | valid_loss: 1.6673 cycle: 0:01:04.658950 total_time: 0:05:14.450527
 40%|███████████████████████████████████▏                                                    | 2000/5000 [06:59<06:25,  7.77it/s]^[[Citer: 2,000 | train_loss: 1.3459    | valid_loss: 1.5890 cycle: 0:01:04.577669 total_time: 0:06:59.165923
 50%|████████████████████████████████████████████                                            | 2500/5000 [08:45<05:25,  7.69it/s]iter: 2,500     | train_loss: 1.2720    | valid_loss: 1.5538 cycle: 0:01:05.944031 total_time: 0:08:45.437820
 59%|████████████████████████████████████████████████████▎                                   | 2973/5000 [10:26<04:21,  7.75it/s] 60%|████████████████████████████████████████████████████▊                                   | 3000/5000 [10:30<04:25,  7.53it/s]iter: 3,000     | train_loss: 1.2085    | valid_loss: 1.5571 cycle: 0:01:04.693232 total_time: 0:10:30.205539
 70%|█████████████████████████████████████████████████████████████▌                          | 3500/5000 [12:16<03:13,  7.74it/s]iter: 3,500     | train_loss: 1.1388    | valid_loss: 1.5596 cycle: 0:01:05.024825 total_time: 0:12:16.216906
 80%|██████████████████████████████████████████████████████████████████████▍                 | 4000/5000 [14:01<02:09,  7.73it/s]iter: 4,000     | train_loss: 1.0722    | valid_loss: 1.6034 cycle: 0:01:05.704443 total_time: 0:14:01.813210
 90%|███████████████████████████████████████████████████████████████████████████████▏        | 4500/5000 [15:49<01:04,  7.72it/s]iter: 4,500     | train_loss: 0.9874    | valid_loss: 1.6838 cycle: 0:01:06.705413 total_time: 0:15:49.758759
100%|███████████████████████████████████████████████████████████████████████████████████████▉| 4999/5000 [17:35<00:00,  7.51it/s]iter: 4,999     | train_loss: 0.8881    | valid_loss: 1.8121 cycle: 0:01:05.292599 total_time: 0:17:35.475523
100%|████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [18:16<00:00,  4.56it/s]
total runtime: 0:18:16.412797
================================================================================
Generative Output!
================================================================================

And this bright face of base King Edward's death.

OXFORD:
Slanders, and pity what I bury, didst the sword,
And, not with true towns at for Edward's life;
Which, if the new orish Hereford the king?

TYBALT:
No, my glad of Hereford, my Lord Northumberland,
When some Henry Sixth God John Lycome Northumbol,--
And the king of Somerset, Each of Warwick!
3 KING HENRY VI

YORK:
What's this, Warwick, I know against Margaret,
The rages of Cretervance and devil Edward.

YORK:
Master, your high-hand follow
(base) root@DESKTOP-605N4AP:/home/tlee/myrepos/course-chat-gpt# 
```