# Bio-Inspired Robotics: MuscleWalker Project
## 25-Minute Presentation Outline

---

### Slide 1: Introduction
*   **Title:** MuscleWalker: Learning Bio-Inspired Locomotion
*   **Presenter:** [Your Name]
*   **Context:** Final Project for Bio-Inspired Robotics
*   **Hook:** "How can we teach a robot to walk not by calculating angles, but by pulling strings like a puppet?"

### Slide 2: Problem Statement
*   **The rigidity of traditional robotics:**
    *   Traditional robots often use stiff, high-torque electric motors.
    *   Leads to unnatural, jerky movements and low energy efficiency.
*   **The "Bio" Gap:**
    *   Biological systems utilize compliant muscle-tendon units that allow for fluid motion and energy storage.
    *   *Problem:* Controlling these highly non-linear, redundant actuation systems is mathematically difficult for traditional control theory.

### Slide 3: Biological Inspiration
*   **Model Organism:** The Human Musculoskeletal System.
*   **Key Features Emulated:**
    *   **Compliance:** Tendons act as springs (Series Elastic Actuators) to store energy, like the Achilles tendon.
    *   **Redundancy:** Multiple muscles for one joint (agonist-antagonist pairs).
    *   **Morphology:** Light legs, heavy torso (low distal mass) for efficient swinging.

### Slide 4: State of the Art - 1X Neo
*   **Industry Example:** **1X Neo** (Humanoid Robot).
*   **Why it matters:**
    *   Unlike Boston Dynamics' Atlas (hydraulic/rigid), Neo is designed for safe human interaction.
    *   Uses "bio-inspired" cable-driven or compliant actuation to move naturally and silently.
    *   *Relevance:* Our project explores the underlying control logic (Reinforcement Learning) that likely powers such compliant bio-robots.

### Slide 5: Method - High Level Overview
*   **Flowchart of Progress:**
    1.  **Modeling:** Created a single tendon-based muscle unit in MuJoCo.
    2.  **Integration:** Built a full bipedal walker model using these tendon actuators.
    3.  **Testing (MPPI):** Used Model Predictive Path Integral (MPPI) to test the model's dynamics and balance capabilities.
    4.  **Training (RL):** Trained different policies (Walk, Trot, Jump) using Reinforcement Learning (PPO) to achieve stable locomotion.

### Slide 6: Muscle Modeling (1/3) - The Concept
*   **Hill-Type Muscle Model:**
    *   Muscles are not just force generators; they are **spring-dampers**.
    *   **Force = Activation × f(Length) × f(Velocity)**.
*   **Implication:**
    *   A stretched muscle pulls harder (passive stability).
    *   A fast-contracting muscle produces less force (damping).

### Slide 7: Muscle Modeling (2/3) - Single Tendon Model
*   **MuJoCo Implementation:**
    *   **Sites:** Attachment points on bones (Origin & Insertion).
    *   **Spatial Tendon:** The path wrapping around the joint.
    *   **Actuator:** The muscle pulls on this tendon.
*   **Key Property:**
    *   **Active Tendon (Flexor):** Low stiffness, controlled by the agent.
    *   **Passive Tendon (Extensor):** High stiffness, acts as a spring to support weight.

### Slide 8: Muscle Modeling (3/3) - The Whole Model
*   **Full Biped Architecture:**
    *   **Pairs:** 6 Agonist-Antagonist pairs (Hip, Knee, Ankle for both legs).
    *   **Total Actuators:** 12 tendons, but simplified to 6 active controls (Agent controls Flexors, Extensors are passive springs).
*   **Result:** A 7-link biped that stands passively but needs active control to walk.

### Slide 9: MPPI (Model Predictive Path Integral)
*   **What is it?** A sampling-based control strategy.
*   **How it works:**
    *   Simulate 1000s of random muscle twitch sequences in parallel.
    *   Evaluate which ones keep the robot upright.
    *   Take the weighted average of the best sequences.
*   **Pros/Cons:** Good for immediate balance, bad for long-term walking (short horizon).

### Slide 10: Reward Function Design
*   **The "Teacher":** The reward function tells the robot *what* to do, not *how* to do it.
*   **Shaping:**
    *   Start simple: "Just don't fall."
    *   Add complexity: "Move forward."
    *   Refine: "Move forward *efficiently* and *symmetrically*."
*   **Structure:** We use a **Multiplicative Reward** ($R = r_{stand} \times r_{move}$) to prevent the robot from "hacking" the score by just standing still.

### Slide 11: Output and Graphs - Interpreting Results
*   **What we look for:**
    *   **Mean Reward:** Does it go up? (Learning is happening).
    *   **Episode Length:** Does it reach the max time? (Robot isn't falling).
    *   **Entropy:** Is it still exploring? (High entropy = random actions, Low = confident).

### Slide 12: RL Algorithm - PPO
*   **Algorithm:** Proximal Policy Optimization (PPO).
*   **Why PPO?**
    *   Robust to noise (and physics simulations are noisy).
    *   Sample efficient enough for our dimension (~17 observations, 6 actions).
*   **Library:** Stable-Baselines3.

### Slide 13: Task 1 - Walk Reward
*   **Goal:** Stable, forward locomotion.
*   **Equation:**
    *   $r_{standing} = \frac{2 \cdot (1 - |height_{err}|) + \frac{\cos(\theta) + 1}{2} + posture_{knee} + posture_{hip}}{5}$
    *   $r_{move} = \text{clip}(v_x, 0, 1)$
    *   $R_{total} = r_{standing} \times r_{move}$
*   **Key Components:**
    *   *Standing:* Height > 1.0m.
    *   *Velocity:* Target speed 1.0 m/s.
    *   *Constraint:* Multiplicative structure ensures $R=0$ if robot stops.

### Slide 14: Task 1 - Walk Graphs
*   *Display TensorBoard screenshot for Walking*
*   **Analysis:**
    *   **Phase 1 (0-500k):** Rapid rise in episode length (learning to balance).
    *   **Phase 2 (500k-2M):** Slow rise in reward (learning to move forward).
    *   **Convergence:** Stable walking achieved around 3M steps.

### Slide 15: Task 1 - Walk Video
*   **(Placeholder for Video/Demo)**
*   *What to observe:*
    *   Compliance in the knees (natural shock absorption).
    *   Arms/Torso balancing (if applicable).

### Slide 16: Task 2 - Trot Reward
*   **Goal:** Faster, rhythmic movement.
*   **Equation:**
    *   $r_{trot} = r_{walk\_standing} \times \text{clip}(\frac{v_x}{1.5}, 0, 1) - c_{control} \cdot ||u||^2$
*   **Key Differences:**
    *   **Target Speed:** Increased to 1.5 m/s.
    *   **Inputs:** Added *Phase* ($\phi$) and *Previous Action* ($u_{t-1}$) to observations.
    *   *Reasoning:* Trotting requires explicit timing and rhythm ($\phi$), not just reactive feedback.

### Slide 17: Task 2 - Trot Graphs
*   *Display TensorBoard screenshot for Trotting*
*   **Analysis:**
    *   Learning curve is steeper (harder task).
    *   Variance is higher (trotting is less stable than walking).

### Slide 18: Task 2 - Trot Video
*   **(Placeholder for Video/Demo)**
*   *What to observe:*
    *   More dynamic "bouncing" gait.
    *   Clear flight phases (briefly airborne).

### Slide 19: Task 3 - Dynamic Jumping Reward
*   **Goal:** Explosive vertical jumps and stable landings (Human Athletics).
*   **Equation:**
    *   $R = w_f \cdot \mathbb{I}(air) + w_t \cdot v_{z,takeoff} + w_s \cdot \text{sync}(L,R) + w_h \cdot z_{height} - w_e \cdot ||u||^2$
*   **Key Terms:**
    *   **Flight ($\mathbb{I}(air)$):** Binary reward for having NO feet on the ground.
    *   **Sync:** Penalty if feet touch the ground at different times.
    *   **Takeoff ($v_{z,takeoff}$):** Bonus for positive vertical velocity at the moment of liftoff.

### Slide 20: Task 3 - Jumping Graphs
*   *Display TensorBoard screenshot for Jumping*
*   **Analysis:**
    *   Oscillating reward (cyclic jumping motion).
    *   High energy cost (explosive movements require high muscle activation).

### Slide 21: Task 3 - Jumping Video
*   **(Placeholder for Video/Demo)**
*   *What to observe:*
    *   Synchronized leg extension (squat-jump mechanics).
    *   Energy storage in the tendons during landing (Plyometrics).

### Slide 22: Discussion & Challenges
*   **The "Local Minima" Trap:**
    *   Robot learned to just "fall forward" to get velocity reward.
    *   *Fix:* Strict early termination if height drops.
*   **Sim-to-Real Gap:**
    *   Real muscles fatigue; our model doesn't (yet).
    *   Real tendons have hysteresis; ours are perfect springs.

### Slide 23: Future Work
*   **Metabolic Cost:** Optimize for "Miles per Calorie" (Cost of Transport).
*   **3D Terrain:** Walking on stairs or slopes.
*   **Hierarchical Control:** Use MPPI for high-level planning ("Go there") and RL for low-level muscle control ("Flex this").

