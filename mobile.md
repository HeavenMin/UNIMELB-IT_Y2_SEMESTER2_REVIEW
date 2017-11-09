---
typora-copy-images-to: ./mobile_image
---

# COMP90018 Mobile Computing Systems Programming (Mo)

#### Review/Mobile

`Author : Min Gao` 

> [Week 1](#L1) – All Slides
>
> [Week 2](#L2 - UX research) – All Slides
>
> [Week 3](#L3 - Mobile Interaction Design) – All Slides
>
> [Week 4](#L4 - Progessing Sensor Data) – All Slides
>
> [Week 5](#L5 - Context & Activity RECOGNITION) – All Slides
>
> [Week 6](#L6 - RFID) – RFID – Slides 1 to 39
>
> [Week 6](#L6 - Mobile Games) - Mobile Games – Slides 1 to 27
>
> [Week 7](#L7 Location Privacy) – Location Privacy – Slides 1 to 34, 43 to 44, 46 to 47
>
> [Week 7](#L7 Mobile GUIs) – Mobile GUIs – Slides 1 to 44
>
> [Week 8](#L8 Wireless Sensor Networks) - Wireless Sensor Networks – Slides 1 to 76
>
> [Week 9](#L9 Mobile Networks) - Mobile Networks – Slides 1 to 37
>
> Week 10 – Advanced Topics – Not Examinable
>
> Week 11 - Not Examinable

## L1

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

 ![6C8BF49E-751E-4A0C-BCA0-D96E90B6BBDC](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/6C8BF49E-751E-4A0C-BCA0-D96E90B6BBDC.png)

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

#### HCD consists of three phases (3i)

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

 ![B3E9A888-3F5C-4CF8-BAF1-196DF59EE997](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/B3E9A888-3F5C-4CF8-BAF1-196DF59EE997.png)

[Shared Value](https://crowdfavorite.com/the-value-of-balancing-desirability-feasibility-and-viability/)

### The Design Process: The Double Diamond

[The design process: what is the double diamond](http://www.designcouncil.org.uk/news-opinion/design-process-what-double-diamond)

Problem __ divergent thinking (<-discover , develop->) __ Solution

​                \\_ convergent thinking (<-define, deliver ->) ___/

* __Discover__ insight into the problem
* __Define__ the area to focus upon
* __Develop__ potential solutions
* __Deliver__ solutions that work

 ![AACB67FE-0337-4EF5-AEC0-C102EB8B42AF](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/AACB67FE-0337-4EF5-AEC0-C102EB8B42AF.png)

```
Brainstorming is a technique that involves __divergent__ thinking.
```

### Interaction Design

[Interaction Design Sketchbook](https://hci.rwth-aachen.de/tiki-download_wiki_attachment.php?attId=797) 

Hunch -> Hack -> Trial -> Idea =>> Design -> Prototype -> Test -> Principles -> Plans -> Product -> Market -> Paradigms 

### Reasearch Methods category

* __Looking__: Observational Methods

  * [Ethnography](https://www.interaction-design.org/literature/book/the-encyclopedia-of-human-computer-interaction-2nd-ed/ethnography) It is a qualitative orientation to research that emphasises the detailed observation of people in natually occurring settings. 强调人在自然发生的设置详细的观察。
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
    * paper prototypes
    * Pads
    * Carousel
    * Wireframes

* __Testing__:

  * Discount usability 可用性
  * Eye tracking
  * Lab or In-The-Wild (Lab is better)

|                     | Pons                                     | Cons                                     |
| ------------------- | ---------------------------------------- | ---------------------------------------- |
| Ethnography         | in depth<br />Better product             | Not cheap<br />Time consuming<br />Observer effect<br />Specific |
| Interview           | Simple<br />Cost cheaper<br />Context    | Forget<br />Time-consuming<br />You have not direct observation<br />Formal |
| Experience sampling | Context timely<br />Cheap<br />Scale up<br />Feeling | Insight<br />Dirty hand<br />Input       |

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

$$
\text{All together} \\
\text{Initial state estimate: } x_o = \begin{pmatrix} x_0\\ y_0 \\ v_0^{(x)} \\ v_0^{(y)}  \end{pmatrix} = \begin{pmatrix} z_0^{(x)}\\z_0^{(y)}\\0\\0 \end{pmatrix} \\
\text{Initial estimate of state error covariance : }\\
P_0 = \begin{bmatrix} \sigma^2&0&0&0\\0&\sigma^2&0&0\\0&0&\sigma_s^2&0\\0&0&0&\sigma_s^2 \end{bmatrix}\\
\text{STEP 1: PREDICT}\\
x_i^{predicted} = \phi_{i-1}x_{i-1}^{corrected} \\
\text{Extrapolate the state error covariance: } \\
P_i^{predicted} = \phi _{i-1} P_{i-1}^{corrected}\phi_{i-1}^{T} + Q_{i-1} \\
\text{STEP 2: MEASURE} \\
\text{Compute the Kalman gain: } \\
K_i = \dfrac {P_i^{predicted}H_i^{T}}{H_iP_i^{predicted}H_i^T + R_i} = H_i^{-1} \dfrac {H_iP_i^{predicted}H_i^{T}}{H_iP_i^{predicted}H_i^T + R_i} \\
R_i \text{ is the uncertainty from the measurement} \\
H_iP_i^{predicted}H_i^{T} \text{ is the uncertainty propagated by the model} \\
\text{ Update extrapolations with new measurements: } \\
x_i^{corrected} = x_i^{predicted} + K_i (z_i - H_i x_i^{predicted}) \\
p_i^{corrected} = (I - K_i H_i ) P_i^{predicted}
$$

1. Make prediction most like value based on dynamic model. 
2. measure use sensor data to correct the prediction. 

### Pros and Cons for Kalman filter

Pros:

* Dynamic model of the system
* no lag
* Tunable trade-off between model and measurement
* Uncertainty estimate

cons:

* Parameters are not intuitive
* Overshooting

#### Kalman and Particle

* Kalman is cheap to run
* Particle is general

| Kalman   | Particle        |
| -------- | --------------- |
| Bayesian | Bayesian        |
| Gaussian | Gaussian or not |
| Linear   | Linear or not   |
| Online   | Online          |

#### Particle Filter

Generate hypotheses -> compute weights -> resample

Pros:

1. General -> because it doesn't make any assumptions 
2. Continuous or discrete values is available 
3. Great result 

Cons:

1. Lots of memory
2. Very slow

---

## L5 - Context & Activity RECOGNITION

### Elements of the user's context

* Location
* User's identity
* Time of the day
* Sound levels
* Light levels
* User's motion

### Context awareness (activities)

> Who What Where When Why

- context __Display__ (e.g. GPS can track where you are)
- contextual __augmentation__ (e.g. google photo use location and time to create augment info)
- __context-aware configuration__ (configure how devices are used)
- __context- triggered actions__ (changing brightness when light changes)
- __contextual adaptation of the environment__ (turn on heating when go to sleep)
- __contextual mediation__ (based on the limited and needs of the context)
- __context-aware presentation__ (e.g. switch Portrait to landscape)

### Characteristics of Human Activity Recognition (HAR) Systems

- __Execution__ (the timing of the execution of the recognition)
  - __online__:  process data in real time. e.g. human-computer interaction
  - __offline__: records the sensor data first. The recognition is performed afterwards. e.g. health monitoring
- __Generalisation__
  - __user independent__: working with everyone
  - __user specific__: single person data. Performance is usually higher than in the user-independent case, but does not generalise as well to other users.
  - __temporal__: both cases.


- __Recognition__
  - __continuous__ : stream of data. The system automatically “spots” the occurrence of activities or gestures in the streaming sensor data. 
  - __segmented__ : The system assumes that the sensor data stream is segmented at the start and end of a gesture by an oracle. 
- __Activities__
  - __periodic__ : Activities or gestures exhibiting periodicity, such as walking, running, rowing, biking, etc. Sliding window segmentation and frequency-domain features are generally used for classification. 
  - __sporadic__ :The activity or gesture occurs sporadically, interspersed with other activities or gestures. Segmentation plays a key role to isolate the subset of data containing the gesture. 
  - __static__ : The system deals with the detection of static postures or static pointing gestures. 
- __System model__
  - __stateless__ : The recognition system does not model the state of the world. 
  - __stateful__: The system uses a model of the environment, such as the user’s context or an environment map with location of objects. 

```
Imagine an app that analyses your shopping lists and suggests at the end of the week recipes with a better nutriotional value based on the ingredients of that list. Check all characteristics that apply to this app.
Offline, User-specific, Periodic
```

### The processing pipeline

* [sensor](https://en.wikipedia.org/wiki/List_of_sensors)

#### Choosing a sensor

* what is it measuring

- Accuracy and precision
- range of the sensor
- resolution or sensitivity of the sensor
- sampling rate or frame rate
- cost

#### Preprocessing

* Interpolation 
* downsampling

#### Segmentation

- sliding window
  - Non-overlapping sliding window
  - overlapping sliding window
- energy based
- additional context sources

#### Feature extraction

- Min
- Max
- Range
- Skewness
- Mean
- Energy
- Kurtosis
- Variance

####Classifiers

- Hidden Markov Models
- Dynamic Time Wraping
- kNN
- AdaBoost
- Support Vector Machines
- Random Forest 

#### Reporting performance

What proportion of the labelled items were correct?

* __Precision__ = True positives / (True positives + False positives)

What proportion of the windows with that activity were recognised?

* __Recall__ = True positives / (True positives + False negatives)

__F1 Score__ = $\dfrac {2}{\dfrac {1}{\text{Recall}} + \dfrac {1}{\text{Precision}}} = 2 \dfrac {\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ 

#### Confusion Matrix

 ![40DDF3D8-FDCE-4CDF-A783-CF5BE42C6399](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/40DDF3D8-FDCE-4CDF-A783-CF5BE42C6399.png)

---

## L6 - RFID

#### RFID technology

__Tag__

* Microchip connected to an antenna 连接天线的微芯片
* can be __passive, semi-passive, active__ 
* No battery: __passive__
* Semni-passive: circuit is battery-powered except communication 除了通信外，电力是电池供电的
* secure
* query tags via radio signals

__Reader__

* Query tags via __radio signals__    无线电信号

> Example: Visa payWave, payPass

#### RFID (radio frequency identification) 

* Reader sends a radio __interrogation signal__ |询问信号
* RFID tag backscatters its ID       |RFID tag 反射 它的ID

- Proximity-based technology: determine the tag location by

  measuring the signal’s time of flight (in theory)    |通过测量信号的飞行时间来确定标签位置（理论上）

Pros:

* Cheap, high volume, large variety
* Long industry experience
* scanning even with high speeds (300km/h)
* no maintenance, simple to manage

Cons:

* No quality of service
* Only passive data acquistion (asymmetric communication)

#### Characteristics

* No line-of sight necessary
* Resist environmental conditions: frost, heat, dirt, …  |抵抗环境条件：霜冻，炎热，灰尘，…
* RFID tags with read & write memory
* Smartcard functionality (JavaCard): cryptographiccomputations for personal contact cards

__Technical Features for RFID__:

* Reader: simultaneous detection of up to 256 tags, scanning of up to 40 tags per second
* Response time of an RFID tage: less than 100 miliseconds

#### Passive RFIDs

* Do not need an internal power source
* Operating power is supplied by the reader

__Features__

3m, can be very small,  very cheap

#### Active RFIDs

* Own power source (battery life expectancy: up to 10 years) 

__Features__

* cost a few dollars
* samll 
* 100m
* combination with sensors
* deployment in more difficlut RF situations(water)
* Tags have tupically a higher scanning reliability

#### RFID Frequencies

* __LF__: low frequency
  * good __penetration__ of materials including water and metal
  * widely adopted
  * __No collision protocol__ available
  * Range: __30cm__
* __HF__: high frequency
  * Provides __anti-collision__ protocols
  * __1m__
* __UHF__ : ultra-high frequency
  * Difficult to penetrate of water and metal
  * range: __3m__
* __Microwave__
  * range: __2m__
  * high data rate

#### EPC (electronic product code)

* unique number to identify an item in the supply chain

__EPC Device classes__

| EPC Class | Definition                           | Programming     |
| --------- | ------------------------------------ | --------------- |
| Class 0   | read only , passive tags             | by manufacturer |
| Class 1   | write once, read many , passive tags | by customer     |
| Class 2   | rewritable passive tags              | Reprogrammable  |
| Class 3   | Semi-passive tags                    | Reprogrammable  |
| Class 4   | active tags                          | Reprogrammable  |
| Class 5   | Readers                              | Reprogrammable  |

#### Anti-collision & Singulation

__Problem__

* RFID tags are simple and __cannot communicate with other tags__
* High probability that __two tags in communication range respond simultaneously__
* __Collision__: response on the same frequency at the same time

__Anti-collision and singulation protocols__

* Algorithms to identify all tags
* __Anti-collision__: trade time for the possibility to query all tags
* __Singulation__: identify (iterate through) individual tags

#### ALOHA Protocol (resend later)

* based on the classical ALOHA protocol
* __“Tag-Talks-First”__ behavior: tag automatically sends its ID (and data) if it enters a power field.
* If a message collides with another transmission, try resending it later after a random period 

__Collision types__: Partial & complete 

__Reducing collisions in ALOHA__

* __Switch-off__
  * After a successful transmission a tag enters the __quiet state__
* __Slow down__
  * __Reduce the frequency__ of tag responses 
* __Carrier sense__
  * no carrier sense possible
  * Use ACK signal of the reader in communication with another tag
  * Reader broadcasts a MUTE command to other tags if it query one tag

Partial overlap leads to maximum throughput of a 18.4%

#### Slotted ALOHA protocol

* __“Reader-Talks-First”__ : use discrete timeslots SOF (__start-of-frame__) and EOF (__end-of-frame__)
* A tag can send only __at the beginning of a timeslot__
* Leads to __complete or no collision__
* Increased maximum throughput of 36.8%
* can early end

#### Frame-slotted ALOHA

* Group __several slots__ into frames
* Only one tag transmission per frame
* Limits frequently responding tags
* Adaptive version: adjust the number of slots per frame

| Protocol            | +                                        | -                                        |
| ------------------- | ---------------------------------------- | ---------------------------------------- |
| ALOHA               | Adapts quickly to changing<br />numbers of tags<br />Simple reader design | Worst case: __never finishes__<br />__Small throughput__ |
| Slotted ALOHA       | __Doubles throughput__                   | Requires synchronization<br />__Tags have to count slots__ |
| Frame-slotted-ALOHA | Avoids frequently responding tags        | Frame size has to be known or transmitted<br />similar to slotted ALOHA |

#### Binary Tree Protocol I

* __DFS__
  * __“Reader-Talks-First”__ behavior: reader __broadcasts__ a request command with an ID as a parameter
  * A sub-tree T is searched by an identifier prefix
  * Only tags with an ID lower or equal respond 
  * An interrogated tag is instructed to keep quiet afterward 
  * __Repeat algorithm until no collision occurs or all tags are quiet__

#### Binary Tree Protocol II

* Each sub-tree T corresponds to an __identifier prefix__
* Reader searches T by sending prefix, interrogating tags for their next bit
  * If all “0” search Left(T)
  * If all “1” search Right(T)
  * If both “0” and “1” search Left(T) and Right(T)

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/screenshot.png)

#### RFID Applications

* E-passports
  * Security risk: __forgery__
* Transportation payment
* Electronic toll collection
  * Security risk: denial of service
* Vehicles : Smart key
* Supply chain & inventory management
* Preventation
* product tracking
  * privacy risk: tracking
* Human implants

#### Status Quo of RFID Systems

* __No authentication__ 认证
  * Readers are blind: if tag does not reply, reader does not know about it
  * Tags are promiscuous and reply to any reader
* __No access control__ 访问控制
  * Malicious reader can link to a tag
  * Malicious tag can spoof a reader
* __No encryption__
  * Eavesdropping possible (especially for the reader)
* __Man-in-the-middle-attack__

#### Privacy Concerns

* Unauthorized surveillance
  * simple RFID tags support no security mechanisms
  * Permanent RFID serial numbers can compromise privacy
* Potential risks
  * Tags in goods might be a potential risk
  * Threat: scanning of assts of high value

#### RFID Tag Privacy

* killing: tag deactivation
* user intervention
* active jamming
* hash-locking
* encrypting
* one time identifiers
* Hiding: blocker tags
* keyless "encryption"
* __Threats of RFID__
  * Cloning
  * Forgery
  * Relabeling

---

## L6 - Mobile Games

### Low-level UI

#### Tasks of a low-level API

* precise control about what is drawn
* control about the location of an item
* Handle basic events such as key presses
* Access specific keys

#### User interfaces versus Games

| UI is event-driven                     | Game is time-driven                      |
| -------------------------------------- | ---------------------------------------- |
| UI is updated in reponse to user input | Run continuously                         |
| Events: pressing a soft key,           | Updates occur with and without user input |

#### Gamle Loop: main Thread

1-> Check for user input -> update game state -> update screen buffer -> flush buffer to display ->1

#### The purpose of a game API

__Overview__:

* Screen buffer
  * __GameCanvas__
    * Dedicated screen buffer
    * supports incremental updates
    * Flush graphics: display contents of the buffer
* key polling
  * Query the status of keys
    * is a key pressed and which key is pressed
    * Duration of a key press
    * are keys pressed simultaneously or repeatedly
* Layers
  * sprites and tiled layers
  * can be visible or invisible
* sprites
  * Definition
    * Figure in 2D that is part of a larger game scene
    * parts can be transparent
    * A sequence of sprites enables animation
  * Animations
    * Frame sequence of a sprite
    * Ordered list of frames to be shown
    * Frames can be omitted, repeated
    * Sprite is n frames
* tiles (small image)
  * Tile is a small image that can be combined with other tiles to larger images
  * 2D games with large background images are composed of tiles
  * A set of tiles is small, little memory required
* __collision detection__
  * Collision rectangle
    - each sprite has a collision rectangle, usually the size of the sprite
  * Boundary-level detection (fast)
    - Test if two collision rectangles intersect
  * Pixel-level detection (precise)
    - collision if opaque(不透明) pixels touch

#### What is different for Mobile Games

> Features of Mobile Game development

* Processing & network
  * less cpu power, no hardware acceleration. less memory, intermittent network connections
* Hardware
  * Input capabilities
  * screen size
* Portability
  * Sensors: locaiton, acceleration, camera
  * Context-awareness, use environment as part of the game 
  * Device as controller
  * Mixed reality games, location-based games

#### Tips for good UI

| Prefer               | Avoid                     |
| -------------------- | ------------------------- |
| Relative positioning | Absolute positioning      |
| Text extensively     | many pictures             |
| compress images      | large images              |
| reduce image size    | Animations (except games) |
| Separate page sets   | Horizontal scrolling      |

### Usability Guidelines for Mobile Games

#### Game Start

* Opening screen
  * splash screen
  * Limit the number of screens before the game start
* Main menu
  * game's main menu: custom graphics
  * Avoid using UI components with standard graphics
  * Help item

#### Game Controls

* General design
  * avoid the need for pressing two key simultaneously: diffucult on a small keyboard
  * gestures
  * one key one command
* In-game design
  * pasue the game and show the main menu
* Pause & save
  * Single-player games
    * provide save game capability
  * Two-player games
    * pause model applies to both players
    * provide information about why the game is paused
  * Multiplayer games
    * interruption of one player does not impact other palyers
    * switch player to background or drop player from the game
* Feedback
  * status information
    * Health, points ,level ,score
    * Not too much technical information
  * Clear feedback on game goals
  * Multiplayer games
    * who has won
* Game Experience
  * Easy to learn, but difficult to master
  * Rewards
  * Difficulty level
* Noise pollution
  * Sound volume
    * Default volume
    * enable different sound levels for background music and game sounds
    * ability to turn sounds off quickly
    * no high-pitched sounds
  * Bluetooth multiplayer games
    * Synchronize the background music
* Distinctive Graphics
  * avoid small text on the screen
  * Appearance of game objects and characters
    * Easily understood
    * different items should look different
  * Multiplayer games need to identify who is who
* Post game
  * High score lists
  * easy restart

#### Criteria for mobile games

* easy to learn
* Interruptible
* Subscription
* social interactions
* take advantage of smartphones

#### optimizing mobile games

Fisrt complete the game, optimize later

90/10 rule

* 90 percent of execution time
* 10 percent of the code
* use a profiler

But aim to improve the actual algorithms before resorting to low-level techniques

#### Optimization trick

* use __stringbuffer__ instead of string
* access class cariables directly
* use local variables
* variables are more efficient than arrays
* count down in loops
* use compound operators
* remove constant calculations in loops
* reuse objects
* assign null to unused objects & unused threads

---

## L7 Location Privacy

> Location-based services & location privacy

__LBS__ : location-based services	|基于位置的服务

* services that integrate a mobile's device location with other information

Mobile operators:

* Voice
* data
* location information

__push v.s. pull__

* user receives information without an active request
* user actively pulls information from the network

#### Applicaiton

* infotainment services 娱乐
* tracking services
* information dissemination(transmission) services 信息传播
* Emergency support systems
* Location-sensitive billing
  * toll payment

#### LBS generations

* 1st generation
  * manual user input of location information
* 2st
  * locaiton information is acquired automatically within a couple of kilometers
* 3st
  * high position accuracy & automatic initiation of services

#### LBS and their required Accuracy

* High accuracy
  * Asset tracking
  * directions
  * emergency
* Medium to high accuracy
  * advertising
  * car navigation
  * POI (point of interest)
* Low accuracy
  * fleet management
  * news
  * traffic information

#### Location Engine

* (reverse) Geocoding 地理编码
  * translate street address to latitude & longitude and vice versa
  * difficult if not complete information available
* Routing & navigation
  * compute best route: A*
* Proximity search
  * POIs such as ATMs

### The important of locaiton privacy

#### Location-based services

* nearest neighbor queries
  * Heart patient need closest hospitals
* monitoring for traffic applications
* Location-aware social networking
  * finding friends
* locaiton-based advertising
  * send coupons to user close to the store

#### Location Privacy

* status of current mobile systems
  * able to monitor,communicate and process information about person's locaiton
  * have a high degree of spatial and temporal precision and accuracy
  * might be linked with other data
* Important research issue
  * we need to protecting location privacy

#### The importance of location privacy

* Location-based spam
* personal safety
* intrusive inferences (people's information)
  * people's view
  * individual preferences
  * health conditons

### Techniques for location privacy

* __Stealth__
  * ability to be at a locaiton without anyone knowing your are there
    * use a passive devices such as GPS
    * cons
      * active devices such as mobile phones cannot preserve stealth
      * access of information overrides stealth
* __Anonymity-based approaches__
  * separate location information from an individual's identity
    * special type is pseudonymity 笔名(nickname)
    * Cons:
      * weak to data mining
      * a barrier to authentication and personalization
* __K-Anonymity__
  * A table is k-anonymous if every record in the table is indistinguishable from at least k − 1 other records with respect to quasi identifiers
* __Anonymity: cloaking__
  * __Spatial cloaking__
    * ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/screenshot-0212004.png)
  * __k-anonymity__
    * Individuals are k-anonymous if their location information cannot be distinguished from k−1 other individuals
  * __temporal cloaking 暂时的隐身__
    * reduce the frequency of temporal information

![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/screenshot-0212052.png)

### Location privacy through Obfuscation  迷惑

> Obfuscation

people are more prepared to reveal their locaiton to some degree the less precise the location is

#### Obfuscation

* mask an individual's precision
* decrease the quality of information about an individual's location
* identity can be revealed
  * assumption
    * spatial imperfection = privacy
    * The greater the imperfect knowledge about a user's location, the greater the user's privacy

![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/screenshot-0212496.png)

#### l-Diversity

* A q*-block is l-diverse if contains at least l "well represented" values for the sensitive attribute S
* A table is l-diverse if every q*-block is l-diverse
* An attacker needs l-1 damaging pieces of background knowledge to eliminate all l-1 possible sensitive values

### Decentralised approach to Location privacy

#### Centralized approaches

![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/screenshot-0212738.png)

__Limitations__

* communication overheads
* security threats
* single point of failure

#### If you do not trust anyone, go decentralized approach

Idea

* Use WPANs
* Clique
* do not disclose your precise position to anyone
* be k-anonymous to your lBS provider

Rules:

* Hide service request from mobile phone operator |, i.e., separate an agent's request (query requestor) from the agent requesting this service (query initiator)

---

## L7 Mobile GUIs

#### Programming mobile devices (fundamental approaches)

* __Server-based approach__
  * create web service
  * client accesses the content via a browser
* __Device-based approach__
  * develop app with an SDK
  * deploy the applicaiton locally on the mobile devices

#### UX Design Priciples

* __Minimize the amount of work__ required for a task
* Acknowledge __limitations of users__
  * only show information that is require
* acknowledge __user mistakes__
  * Anticipate and prevent them
* Acknowledge __human memory__
  * do not rely on human memory
* Various __tidbits__ 花边
  * decide whether to stand out in terms of being different or novel or if a task has to avoid distraction

#### The life cycle of a MIDlet

![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/screenshot-0213385.png)

#### MIDP GUI Programming

* MIDP vs AWT
  * AWT is designed for PCs
  * AWT for mouse
  * AWT supports window managements
* __smartphones__ have different requirements
  * a single screen
  * no overlapping windows ( no window manager required)
  * no complex tables
  * input often limited to keypad or virtual keyboards

#### Input mechanisms in mobile

* keypad
* keyboard
* Pen-based input
* voice input

#### UI support

The difference in mobile like __screen size, screen orientation, input capabilities__

__Abstraction__

* use abstarct descriptions: provide a "Cancel" button
* less code in your applicaiton

#### High-level user interfaces

Goal: __portability __  为了多平台共用方便

* high degree of abstraction
* no dedicated (focus) control of look and feel
* applicaiton uses native look and feel
* good end-user experience

consequences:

* drawing is performed by the OS of the device
* navigation & low-level functions are done by the device

#### low-level user interfaces

Goal: __precise control and placement__  为了让用户使用更方便

* Games, charts, graphs
* control of what is drawn on the display
* handel events such as key presses and releases
* access specific keys

consequences for portabilty

* Platform-independent methods   这样设计不同平台就得单独设计
* Discover the size of the display, orientation, other capabilities

#### (MIDP) GUI Guidelines

| Ensure portability across different devices | KISS principle                      |
| ---------------------------------------- | ----------------------------------- |
| Use high-level API                       | Simple and easy to use UI           |
| Use platform independent parts of low-level API | Minimize user input and offer lists |
| Discover screen resolutions              | Pre-select likely choices           |

#### UI Elements

* Buttons
  * intercpets touch events and sends an action msg to a target object when tapped
* Checkboxes & radio buttons:
  * select one or more options from a set
* Switches
  * On/off buttons
* Segmented controls
  * horizontal control made of multiple segments
* Stepper
  * {+ -}
* Slider
  * 滑动条 select a single value
* Popup menus
  * menu interface
* Pickers
  * pick a time or pick a date
* UITextField
  * Displays editable text and sends an action msg to a target object when user presses the return button
* UITextView
  * Support a scrollable, multiline, editable text region for larger texts
* UILable
  * Implements a read-only text view
* TextView
  * displays text to the user and optinally allows them to edit it
* EditTExt
* Attributes
  * URLs and email addresses are 
* Images
  * a view-based container for displaying either a single image or animating images
* Lists
  * means for displaying and editing hierarchical lists of informtion
* Alerts & Dialogs
  * display an alrt message to the user
* Collections
  * oedered collection of data items and presents them using customizable layouts
* Scroll views
* Navigation
* Refresh

![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/screenshot-0214569.png)

---

## L8 Wireless Sensor Networks

> routing in WSNs

Wireless advantage: all neighbors in broadcasting range listen

__Unit Distance Graph__ (UDG)

* communication range is same for all nodes
* Connectivity graph of nodes: nodes are connected if their  normalized distance is less than 1 

#### Communication

* Communication is expensive (use only if necessary)
* A node communicates only with its neighbors
* wireless advantage: all negibors broadcasting range listen

#### Criteria for routing algorithms (SUQ)

* __Size__ of the routing table
* __Quality__ of the route for a given destination
* __Update cost__

> MANET在计算机技术方面定义为移动自组网络的工作组，全称为Mobile Ad-hoc NETworks。
>
> WSN wireless sensor network

Wireless Sensor networks are MANETs, but with

* larger nodes
* fault tolerance : sensor nodes are more prone to failure
* energy-awareness : nodes have a very limited amount of energy

#### Terminology

* __Routing__
  * Transport msg between two nodes
* Data dissemination
  * transport msg from a node to __many nodes__
* Broadcasting
  * transport msgs from a node to all nodes
* Data gathering
  * Transport msg from nodes to a __sink__
* Base station (BS)
  * node providing a gateway or central processing
* __Sink__
  * Node requesting information
* Source
  * node generating information (event)
* __Interest__
  * message requesting a certain type of information

#### Sensor network architectures

* __Layered__ architecture
  * a single powerful base station with layers
  *  ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/screenshot-0215407.png)
* __Flat__ architecture
  * each node has the same role
* __Hierarchical or clustered__ architecture
  * nodes are organized into clusters
  * nodes sends msg to cluster heads CHs
  * CHs send their msg to base station (BS)
  *  ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/screenshot-0215477.png)

### Topology-based Routing (3 type)

> use information about links in the network

* __proactive__ protocols
  * compute routes before routing
* __reactive__ protocols
  * discover routes on-demand
* __hybrid__ protocols
  * compute routes once, then update

#### Flooding

* Each node that receives a message __broadcasts__ this message if
  * the node is not a goal node
  * the maximum hop count is not reached
* It's a __reactive__ protocol
  * requires no topology maintenance
  * no route discovery necessary
  * often used as __backup strategy__

Cons: 

* __Implosion__
  * a node often receives the same msg from different neighbors
* Duplication
  * Nodes send the same msg to their neighbors
* __Resource blindness__
  * not aware of the energy levels of the mobile device

#### Gossiping

* Limited broadcast
  * nodes do not broadcast received msg to every neighbor but only to a __randomly__ selected neighbor
* pros
  * No __implosion__ and __lower overhead__
* Cons
  * __long travel time__ for msg
  * __no delivery guarantee__

#### Locality insensitivity (Radius Growth)

Destination is a few hops away but the entire network is flooded.

__solution__: Flood with growing radius; using slow flooding

* a timeout for nodes before a message is forwarded

#### Source routing

> Node store routing information for other nodes

* source node stores the whole path to the destination
* source node encodes the path with every message
* nodes on the path remove their ID from the message before relaying the message to the next node.

#### Dynamic source routing

> improving source routing

* caching of routes
* Local search: flooding with TTL+1
* Hierarchy of nodes: nodes with the same IP prefix are in the same direction
* CLustering
* Implicit acknowledgement: symmetric links

#### Directed Diffusion (定向扩散)

`TODO`

#### flooding

* Query flooding
  * A node __interested__ in an event floods the network  (message requesting a certain type of information)
  * Transmission for n nodes: Q * n (Q is the number of Queries)
* Event flooding
  * A node __sensing__ an event floods the network
  * Transmission for n nodes : E * n (E is the number of events)

#### Rumor Routing

* Agent-based algorithm


* * A compromise between query flooding and event flooding
    * spread information from both
    * use only linear paths to preserve energy
  * long-lived msg
  * agents inform other sinks about events
  * routes are not optimal

Cons:

* No delivery gurantee
* Performance depends to topology

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/mobile_image/screenshot-0227936.png)

#### Sample attack scenario

* Pretend to be close to the sink
* attract many packets
* Drop some or all of them
* DoS attacks for geographic routing protocols

`TODO`

---

## L9 Mobile Networks

> Analog signal	模拟信号
>
> carrier signal 	载波信号
>
> digital signal

#### benefits of digital networks

* efficiency
* security
* Degradation and restoration
* Error detection

#### Switching Mechanisms

* Circuit Switching
  * establish a physical connection between sender and receiver
* packet swithcing
  * share a connection between users

#### Multiplexing in 4 dimensions

* Space
* Frequency
* Time
* Code





#### Wireless personal Area networks

* WPAN
  * Bluetooth
  * ZigBee
* WLAN
* WWAN
* satellite

#### WPANs

> wireless personal area networks

* short ranges
* low power consumption
* low cost
* small networks

#### Bluetooth

__Goal__

* Ad-hoc wireless connectivity for electronic devices
* Low-cost replacement for wires

__Radio Technology__

* Short-range: 10m-100m
* Unlicensed ISM frequency band
* 1 mW transmission power
* 2 Mbps

__Networking__

* point to point
* point to multipoint: ad-hoc networking of up to 8 devices
* one device acts as a master, the other devices as slaves

#### ZigBee

__Goal__

* wireless standard for sensing and control applications
* highly reliable and secure, interoperable

__ZigBee__

* Extremely low power
* 200 kbps maximum
* sensors, interactive toys, remote controls
* have Routing protocol : AODV

#### Comparison ZIgBee vs Bluetooth

| ZIgBee                                   | Bluetooth                             |
| ---------------------------------------- | ------------------------------------- |
| Smaller packets over a large network:2^64 | larger packets over a small network:8 |
| low memory requirement: 4-32KB           | require more system resources: 250KB  |
| Rapid network joins in milliseconds      | Long network joins in seconds         |
| Very Low cost: less than one dollar      | Complex design                        |
| Small bandwidth                          | Medium bandwidth                      |
| Medium range:10-100m                     | Medium range 10m (up to 100m)         |
| Battary lifetime: years                  | Battery lifetime: days                |

Bluetooth 比 ZigBee 网络小，需求资源多，长加入时间，复杂设计，稍高带宽，相同range， 更少电力时间



#### other:

9.1.6 Multiplexing

\1. Idea
• Multiple channels share one medium with minimal interference

\2. Multiplexing in 4 dimensions

• Space
• Frequency
• Time
• Code

\3. Guard spaces
• Reduce risk of interference between channels

\4. Space Division Multiplexing

(a) SDM
• Communication using a single channel
• Space channels physically apart to avoid interference

(b) Cellular network
• Reuse frequencies if certain distance between the base stations

• How often can we reuse frequencies?• Graph colouring problem

37

\5. Frequency Division Multiplexing(a) FDM

• Division of spectrum into smaller non-overlapping frequency bands with guard spacesbetween to avoid overlapping

• Channels gets band of the spectrum for the whole time(b) Advantages

• No dynamic coordination required

• Can be used for analog networks(c) Disadvantages

• Waste of bandwidth if traffic distributed unevenly• Guard spaces

\6. Time Division Multiplexing

(a) TDM
i. A channel gets the whole spectrum for a short time

ii. All channels use same frequency at different timesiii. Co-channel interference: overlap of transmissions

(b) Advantage
• High throughput for many channels

(c) Disadvantage
• Precise clock synchronisation

\7. Combining FDM & TDM

(a) FDM/TDM
• Each channel gets a certain frequency band for a certain amount of time
• More efficient use of resources

(b) Advantage
• Higher data rates compared to CDM
• More robust against interference and tapping

(c) Disadvantage
• Precise clock synchronisation

\8. Code Division Multiplexing

(a) CDM
• Each channel has a unique code (chipping sequence)

• All channels use the same frequency
• Each code must be sufficiently distinct for appropriate guard spaces
• Uses spread spectrum technology

(b) Advantages
• No coordination and synchronisation required
• Bandwidth efficient

(c) Disadvantage
• Lower user data rates





















---

