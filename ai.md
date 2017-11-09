---
typora-copy-images-to: ./ai_image
---

# COMP90054 AI Planning for Autonomy(AI)

#### Review/AI

`Author : Min Gao`

```
#Word list
heuristic    启发式，探索的
BTW          by the way
rationality  理性的
omniscience  全知
mainstream   主流
impasse      僵局
perceive     感觉，发觉
percepts     认知
actuator     执行者
softbot      软件机器人
hypothetical 假设的
dissertation 学位论文
impediment   妨碍
Exploration    探索
Exploitation    利用
```

## L-1

### Artificial intelligence: 

* engineering: 1. The science of making machines do things that would require intelligence if done by humans.
* another: 2. The exciting new effort to make computers think. Machines with minds, in the full and literal sense.
* rational: 3. The branch of computer science that is concerned with the automation of intelligent behavior.

从环境中感知信息并执行行动的agent的研究。

#### Intelligent behavior

make ‘good’ (rational) action choices

#### Agents

perceive the environment through sensors -> percepts

act upon the environment through actuators -> actions

__Omniscient__ agent knows everything about the environment, and knows the actual effects of its actions

__Rational__ agent just makes the best of what it has at its disposal, maximizing expected performance given its percepts and knowledge.

__mapping input to the best possible output__:

performance measure * percepts * knowledge -> action

|          | Humanly                                  | Rationally                               |
| -------- | ---------------------------------------- | ---------------------------------------- |
| Thinking | Systems that think like humans<br />(cognitive science)<br />像人一样思考 | Systems that think rationally<br />(Logics: Knowledge and Deduction)<br />合理地思考 |
| Acting   | Systems that act like humans<br />(Turing Test)<br />像人一样行动 | Systems that act rationally<br />(How to make good action choices)<br />合理地行动 |

A central aspect of intelligence is the ability to act successfully in the world

#### solver example:

problem => solver => solution

__solver__ is general as deals with any problem expressed as an instance of model linear equations model, however, is tractable, AI models are not…

#### AI Solver

basic models and tasks include:

* __Constraint satisfaction/SAT__ : find state that satisfies constraints
* __Planning problems__: find action sequence that produces desired state
* __Planning with Feedback__ : find strategy for producing desired state





















___

