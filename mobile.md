---
typora-copy-images-to: ./stream_image
---

# COMP90018 Mobile Computing Systems Programming (Mo)

#### Review/Mobile

`Author : Min Gao`

```
Week 1 – All Slides
Week 2 – All Slides
Week 3 – All Slides
Week 4 – All Slides
Week 5 – All Slides
Week 6 – RFID – Slides 1 to 39
Week 6 - Mobile Games – Slides 1 to 27
Week 7 – Location Privacy – Slides 1 to 34, 43 to 44, 46 to 47
Week 7 – Mobile GUIs – Slides 1 to 44
Week 8 - Wireless Sensor Networks – Slides 1 to 76
Week 9 - Mobile Networks – Slides 1 to 37
Week 10 – Advanced Topics – Not Examinable
Week 11 - Not Examinable
```

## L1– All Slides

### Technical Constraints

1. Disconnections
2. Bandwidth
3. Latency
4. Network heterogeneity
5. Security
6. Energy Usage
7. Risk to data
8. Memory
9. Processing power
10. Input/Output

### Mobile application types

1. Native
2. Web
3. Hybrid

{speed, sensors, dev resources, dev speed, ux/ui, data transfer. updating, offline, monetisation(货币化)}

__Mobile computing__ is about __Devices__, __Users__, __Agents__

### Mobile device Types

* Luggable
* Portable
* Wearable
* Insertable

### Design fiction

> The deliberate use of diegetic prototypes to suspend disbelief about change

* something that creates a story world
* has something prototyped in that story world
* does so in order to create a discursive space

---

## L2 - UX research

### UX

UX {User Experience Design} [Usability: A part of the User Experience](https://www.interaction-design.org/literature/article/usability-a-part-of-the-user-experience)

 ![6C8BF49E-751E-4A0C-BCA0-D96E90B6BBDC](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/stream_image/6C8BF49E-751E-4A0C-BCA0-D96E90B6BBDC.png)

#### HCI

HCI {Human Computer Interaction} [人机交互，交互设计](https://www.zhihu.com/question/20027517)

#### Don Norman 三要素

[Don Norman 三要素](https://www.ted.com/talks/don_norman_on_design_and_emotion?language=zh-tw)

1. __Visceral__ Design (本能层设计): The subconscious reaction based on appearance. {Look, Feel, Sound}
2. __Behavioral__ Design (行为设计): The reaction that stems from the ease of difficulty of use. {Function, Understandability, Usability, Physical feel}
3. __Reflective__ Design (反思性设计): The reaction that derives from self-image, experience and memories.

[《情感化设计》的笔记-本能，行为，反思](https://book.douban.com/annotation/16472560/)

[情感设计中本能 、 行为 、 反思的解析与表达](https://wenku.baidu.com/view/2a7ee027f90f76c660371a7a.html)

```
What are the three levels of processing that Don Norman proposes?
1. Visceral	2. Behavioral	3. Reflective
```

#### How do we know what to build

Understand the domain $\rightarrow$ Design $\rightarrow$ Build the App $\rightarrow$ Evaluation $\rightarrow$ _Understand the domain_

###  HCD

{Human Centered Design} 以人为本的设计

HCD is a creative approach to problem solving and the backbone. It’s a process that starts with the people you’re designing for and ends with new solution that are tailor made to suit their needs. HCD is all about building a deep empathy with the people you’re designing for;generating tons of ideas;building a bunch of prototypes;sharing what you’ve made with the people you’re designing for; and eventually putting your innovative new solution out in the world.

#### HCD consists of three phases

* __Inspiration__ : you’ll learn directly from the people you’re designing for as you immerse yourself in their lives and come to deeply understand their needs
* __Ideation__ : you’ll make sense of what you learned, identify opportunities for design, and prototype possible solutions.
* __Implementation__ : you’ll bring your solution to life, and to market. And you’ll know that your solution will be a success because you’ve kept people you’re looking to serve at the heart of the process.

#### Shared Value 共同价值

* __Desirability__ 愿望，客观需要 (Wishes of the stakeholder) — __Do they want this?__
  * Will this fill a need
  * Will they actually want it
  * Will this fit into their lives
  * Will it appeal to them
* __Feasibility__ 可行性 (Your ability to deliver) — __Can we do this?__
  * Is the tech within reach
  * How long will this take
  * Can we make it happen
* __Viability__ 行可能性 (whether you should start the project) — __Should we do this?__
  * Will this align with business goals
  * Does it fit the client’s budget
  * What is the ROI/Opportunity cost 

 ![B3E9A888-3F5C-4CF8-BAF1-196DF59EE997](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/stream_image/B3E9A888-3F5C-4CF8-BAF1-196DF59EE997.png)

[Shared Value](https://crowdfavorite.com/the-value-of-balancing-desirability-feasibility-and-viability/)

### The Design Process: The Double Diamond

[The design process: what is the double diamond](http://www.designcouncil.org.uk/news-opinion/design-process-what-double-diamond)

Problem __ divergent thinking (<-discover , develop->) __ Solution

​                \\_ convergent thinking (<-define, deliver ->) ___/

* __Discover__ insight into the problem
* __Define__ the area to focus upon
* __Develop__ potential solutions
* __Deliver__ solutions that work

 ![AACB67FE-0337-4EF5-AEC0-C102EB8B42AF](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/stream_image/AACB67FE-0337-4EF5-AEC0-C102EB8B42AF.png)

```
Brainstorming is a technique that involves __divergent__ thinking.
```

### Interaction Design

[Interaction Design Sketchbook](https://hci.rwth-aachen.de/tiki-download_wiki_attachment.php?attId=797) 

Hunch -> Hack -> Trial -> Idea =>> Design -> Prototype -> Test -> Principles -> Plans -> Product -> Market -> Paradigms 

### Reasearch Methods category

* __Looking__: Observational Methods

  * [Ethnography](https://www.interaction-design.org/literature/book/the-encyclopedia-of-human-computer-interaction-2nd-ed/ethnography) It is a qualitative orientation to research that emphasises the detailed observation of people in natually occurring settings.
  * [Shadowing](https://www.interaction-design.org/literature/article/shadowing-in-user-research-do-you-see-what-they-see) The researcher accompanies the user and observes how they use the product or service within their natual environment.
  * [Video Diary](https://www.nngroup.com/articles/diary-studies/) It collect quantitative data which is self-reported by participants over a period.
  * [Case Study](http://dl.acm.org.ezp.lib.unimelb.edu.au/citation.cfm?id=2994146)

* __Learning__: Information gathering

  * [Cultural Probes](https://www.zhihu.com/question/21938802)  文化探针研究

  * Technology Probes (the tech probes draw from __Social Science__, __Design__ and __Engineering__) 

  * Competitor Analysis (对竞争者应用的研究)

  * ```
    How are tech probes different to other prototypes or prodcut?
    1. They should only contain a handful of functions
    2. They should be instrumented to collect data from users
    3. They should be open for interpretation
    ```

* __Asking__: talking to people

  * [Questionnaires](https://www.nngroup.com/articles/qualitative-surveys/)
  * [Interviews](https://www.nngroup.com/articles/interviewing-users/)
  * Contextual Inquiry [情境调查](https://www.sitepoint.com/contextual-enquiry-primer/)
  * Guessability
  * [Experience Sampling](https://pdfs.semanticscholar.org/23c1/a30e53e2e254cf642e48ee6e3fcd1de1f1a5.pdf) alerts
    * Random
    * Scheduled
    * Event-based

* __Prototyping__: learning by building

  * Sketching  [📚: Sketching User Experiences & Sketching user experiences The workbook]
  * Task Flows {Tools: [Marvel](https://marvelapp.com/sketch/), [balsqmiq](https://balsamiq.com), [sketch](https://www.sketchapp.com)}
    * Pads
    * Carousel
    * Wireframes

* __Testing__:

  * Discount usability 可用性
  * Eye tracking
  * Lab or In-The-Wild (Lab is better)

### How to choose a method

[research methods](https://www.nngroup.com/articles/which-ux-research-methods/)

1. What people do
2. what people say
3. Qualitative beats Quantitative

---

## L3 - Mobile Interaction Design

### Waves of Mobile computing

> [Waves of Mobile computing](https://www.interaction-design.org/literature/book/the-encyclopedia-of-human-computer-interaction-2nd-ed/mobile-computing)

* __Portability__: Reducing the hardware size to make it possible to move them relatively easily
  * example: {Dynabook 1968}
* __Miniaturization__: Radically smaller, enabling usage on the move
  * example: {apple Newton 1992}
  * example: {palm pilot 1997}
* __Connectivity__: wireless data networks on the move
  * WAP: wireless application protocol
* __Convergence__: bringing device types together
* __Divergence__: specialised functionality rather than generalised ones
* __Apps__: There’s an app for that
  * [Natural User Interfaces Are Not Natural](https://www.jnd.org/dn.mss/natural_user_interfa.html) 
* __Digital ecosystems__: Pervasive and interrelated technologies working together

### Traditional phone VS Current phone

| Traditional phone | Current phone         |
| ----------------- | --------------------- |
| Tactile           | Touch                 |
| Indirect          | Direct                |
| Key presses       | Gestures              |
| Single key press  | Multi-touch           |
| Portrait          | Portrait or landscape |

### 4C Framework for (Digital ecosystems)

- __Communality__: Sequential interaction involving several users.
  - Personalization 
  - Generalization 
- __Collaboration__: Simultaneous interation by many users.
  - Division:
  - Merging:
- __Continuity__: Sequential interaction involving several artifacts.
  - Sychronization:
  - Migration
- __Complementarity__: Simultaneous interaction with multiple artifacts.
  - Extension:
  - Remote Control:

|                        | Many Users                               | Many artifacts                           |
| ---------------------- | ---------------------------------------- | ---------------------------------------- |
| __sequential__ (序列的)   | __Communality__ (集体性)<br />* __*Personalization*__<br /> The artifact is personalized to each user {like Facebook} <br />个性化服务<br />* __*Generalization*__ <br />The artifact allows different users to immediately interact without any knowledge of their identity {like a ticket machine}<br />公众一致化服务 | __Continuity__ (连续性)<br />* __*synchronization*__<br />a user’s data is synchronized across artifacts {like “cloud-based” storage services like Dropbox}<br />在多个设备之间同步<br />* __*migration*__<br />the state of a user’s activity is migrated from one artifact to another {like Apple’s AirPlay}<br />将正在进行的内容无缝转移到另个设备 |
| __simultaneous__ (同时的) | __Collaboration__ (合作性)<br />* __*division*__<br />the artifact is divided into independent sections and each user can interact with it simultaneously {like split- screen views}<br />同屏幕分屏各自体验<br />* __*merging*__<br />the artifact merges all users’ interaction {like a multi-user board game}<br />同屏不分屏多人体验 | __Complementarity__ (互补性)<br />__*extension*__<br />* interaction with an artifact is extended to another {Adobe’s Nav App moves selected tools in Photoshop onto an iPad}<br />用另个设备来扩展操控一个设备<br />* __*remote control*__<br />one artifact is used to control another {control Apple TV with a app in iphone}<br />纯粹用另个设备当遥控器来远程操控 |

Case Study

* smart phone with smart watches
* phoneTouch

#### Augmenting the interaction with SENSORS

* Orientation sensor
* Flexcase
* table + pen
* Orbits (eye tracking)

---

## L4 - Progessing Sensor Data

### Challenges in Sensor Data

Most data will be like time serious, it's a funciton of time

1. The value of sensor data in time t will strongly influence sensor at time t+1. => GPS 
2. Every sensor has sampling rate -> the data is disparate -> no idea of what happens between sampling rate. 
3. Don't have enough data point. 
4. Remove noise -> may lose some information of data. 

### Filter

> Mean and Median Filters, Kalman Filters, Particle Filters

$z_i = x_i$

$z_i$ is the measurement from sensor, $x_i$ is the actual value on ground truth

But the measurement may not equal to actual value

Therefore, we assume $z_i = x_i + v_i$

$v_i$ is sensor noise, $v_i \sim N(0, \sigma)$ 
$$
z_i = \left(^{z_i^{(x)}}_{z_i^{(y)}}\right) \text{ measurement} \\
x_i = \left(^{x_i}_{y_i}\right) \text{ actual value} \\ 
v_i = \left(^{v_i^{(x)}}_{v_i^{(y)}}\right) \sim \left( ^{N(0, \sigma)} _{N(0, \sigma)} \right) \text{  Normal Dist} \\
$$


#### Mean Filter

Create the window of a given size, wait till window full, each value will be replaced by the mean value in the given window. 

Three problems: 

* wait till window full -> delay of the computation.
* No sharp edges
* easy susceptible to outliers

__Basic uniform Mean__: if window size = 10. the weight that each value will have 0.1

**Weighted Mean Filter**: The old measurement still has effect, the newer, the larger effect.

#### Median Filter

Except for outliers. computing/pick the median. Always use the value exist on the window -> doesn't create the data. good for have the sharp edge (but if the window numbers are too much, the sharpness will be lost).

* less susceptible to outliers
* still need wait till window full
* Does not make up data

#### Both pros and cons for mean/median filter

Pros:

* easy to implement
* efficient
* great cost benefit

Cons:

* Laggy
* No dynamic model

trajectory smooth

### Kalman FIlter

>  underlying velocity

$$
z_i = \binom {z_i^{(x)}}{z_i^{(y)}} \\
x_i = \begin{pmatrix} x_i\\ y_i \\ v_i^{(x)} \\ v_i^{(y)}  \end{pmatrix}  \text{ contain posiitons and velocities total 4 dimensions}\\
\text{Therefore } z_i = Hx_i + v_i \\
H = \begin{bmatrix} 1&0&0&0\\ 0&1&0&0 \end{bmatrix} \\
v_i \sim N(0, R_i) \\
R_i = \begin{bmatrix} \sigma^2&0\\ 0&\sigma^2 \end{bmatrix} \\
$$

#### modelling the dynamics

$$
x_i \leftarrow x_{i-1} \\
\because x_i = x_{i-1} + v_{i-1}\Delta t  \text{ and } v_i = v_{i-1}\\
\begin{pmatrix} x_i\\ y_i \\ v_i^{(x)} \\ v_i^{(y)}  \end{pmatrix} = \begin{bmatrix} 1&0&\Delta t&0\\ 0&1&0&\Delta t \\ 0&0&1&0 \\ 0&0&0&1\end{bmatrix} \begin{pmatrix} x_{i-1}\\ y_{i-1} \\ v_{i-1}^{(x)} \\ v_{i-1}^{(y)}  \end{pmatrix} + Noise \\ 
\begin{pmatrix} x_i\\ y_i \\ v_i^{(x)} \\ v_i^{(y)}  \end{pmatrix} = \begin{bmatrix} 1&0&\Delta t&0\\ 0&1&0&\Delta t \\ 0&0&1&0 \\ 0&0&0&1\end{bmatrix} \begin{pmatrix} x_{i-1}\\ y_{i-1} \\ v_{i-1}^{(x)} \\ v_{i-1}^{(y)}  \end{pmatrix} + \begin{pmatrix} 0\\0\\ N(0, \sigma_s)\\ N(0, \sigma_s)  \end{pmatrix}  \\
\phi_{i-1} = \begin{bmatrix} 1&0&\Delta t&0\\ 0&1&0&\Delta t \\ 0&0&1&0 \\ 0&0&0&1\end{bmatrix} \\
w_{i-1} = \begin{pmatrix} 0\\0\\ N(0, \sigma_s)\\ N(0, \sigma_s)  \end{pmatrix} \\
w_{i-1} \sim N(0, Q_i) \\
Q_i =  \begin{bmatrix} 0&0&0&0\\ 0&0&0&0\\ 0&0&\sigma^2&0 \\ 0&0&0&\sigma^2\end{bmatrix} \\
\text{Therefore, Measurement matrix is : } H =  \begin{bmatrix} 1&0&0&0\\ 0&1&0&0 \end{bmatrix} \\
\text{Measurement noise is : } v_i \sim N(0, R_i) \\
\text{Therefore, Dynamics of  states is : } \phi_{i-1} = \begin{bmatrix} 1&0&\Delta t&0\\ 0&1&0&\Delta t \\ 0&0&1&0 \\ 0&0&0&1\end{bmatrix} \\
\text{Dynamic noise is : } w_{i-1} \sim N(0, Q_i) \\
$$



1. Make prediction most like value based on dynamic model. 
2. measure use sensor data to correct the prediction. 

Dynamic model of the system No lag Tunable trade-off between model and measurement. Uncertainty estimate parameters are not intuitive Overshooting



Bayesian, Gaussian, Linear, Online

**Particle Filter**

Generate hypotheses -> compute weights -> resample

pros:

1. General -> because it doesn't make any assumptions 
2. Continuous or discrete values is available 
3. Great result 

cons:

1. Lots of memory
2. Very slow































---

