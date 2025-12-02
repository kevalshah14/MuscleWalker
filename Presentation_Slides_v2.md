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
    *   Servo motors are stiff, heavy, and inefficient for dynamic walking.
    *   They require precise kinematic planning (inverse kinematics).
*   **The "Bio" Gap:**
    *   Animals don't calculate ZMP (Zero Moment Point).
    *   Animals use compliant tendons to store energy and handle uneven terrain.
    *   *Problem:* Controlling these soft, non-linear muscle actuators is mathematically difficult for traditional control theory.

### Slide 3: Biological Inspiration
*   **Model Organism:** The Bipedal Mammal (Human / Kangaroo).
*   **Key Features Emulated:**
    *   **Compliance:** Tendons act as springs (Series Elastic Actuators).
    *   **Redundancy:** Multiple muscles for one joint (agonist-antagonist pairs).
    *   **Morphology:** Light legs, heavy torso (low distal mass) for efficient swinging.

### Slide 4: State of the Art - 1X Neo
*   **Industry Example:** **1X Neo** (Humanoid Robot).
*   **Why it matters:**
    *   Unlike Boston Dynamics' Atlas (hydraulic/rigid), Neo is designed for safe human interaction.
    *   Uses "bio-inspired" cable-driven or compliant actuation to move naturally and silently.
    *   *Relevance:* Our project explores the underlying control logic (Reinforcement Learning) that likely powers such compliant bio-robots.

### Slide 5: Method - High Level Overview
*   **Simulation Engine:** MuJoCo (Physics with contact dynamics).
*   **The Loop:**
    1.  **Environment:** Muscle-driven biped.
    2.  **Agent:** Neural Network (PPO) or Planner (MPPI).
    3.  **Action:** Muscle Activation [0, 1].
    4.  **Feedback:** New State + Reward.

### Slide 6: Muscle Modeling (1/3) - The Concept
*   **Hill-Type Muscle Model:**
    *   Muscles are not just force generators; they are **spring-dampers**.
    *   **Force = Activation × f(Length) × f(Velocity)**.
*   **Implication:**
    *   A stretched muscle pulls harder (passive stability).
    *   A fast-contracting muscle produces less force (damping).

### Slide 7: Muscle Modeling (2/3) - Implementation
*   **MuJoCo XML Setup:**
    *   **Sites:** Attachment points on bones.
    *   **Spatial Tendons:** Defined paths that wrap around joints (e.g., over the knee cap).
    *   **Moment Arms:** The leverage changes as the joint moves (e.g., pulling the knee is easier when it's bent).

### Slide 8: Muscle Modeling (3/3) - Actuation Dynamics
*   **Active vs. Passive:**
    *   **Flexors (Active):** Low passive stiffness, high control authority. Used to swing the leg.
    *   **Extensors (Passive/Active):** High passive stiffness (stiff springs).
    *   *Function:* Support the robot's weight against gravity without using energy (passive standing).

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
*   **Formula:**
    *   $R = (Standing \times Upright \times Posture) \times Velocity$
*   **Key Components:**
    *   *Standing:* Height > 1.0m.
    *   *Velocity:* Target speed 1.0 m/s.
    *   *Constraint:* If Velocity = 0, Reward = 0 (forces movement).

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
*   **Changes:**
    *   **Target Speed:** Increased to 1.5 m/s.
    *   **Inputs:** Added *Phase* (sinewave) and *Previous Action* to the neural network.
    *   *Reasoning:* Trotting requires timing and rhythm, not just reaction.

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

### Slide 19: Task 3 - Jump (Kangaroo) Reward
*   **Goal:** High vertical hops with synchronous legs.
*   **The Formula:**
    *   $R = r_{flight} + r_{takeoff} + r_{sync} + r_{height}$
*   **Key Terms:**
    *   **Flight Bonus:** Big reward for having NO feet on the ground.
    *   **Contact Sync:** Penalty if feet touch the ground at different times.
    *   **Takeoff:** Bonus for vertical velocity *at the moment* of leaving the ground.

### Slide 20: Task 3 - Jump Graphs
*   *Display TensorBoard screenshot for Jumping*
*   **Analysis:**
    *   Oscillating reward (jumping is cyclic).
    *   High energy cost (jumping takes effort).

### Slide 21: Task 3 - Jump Video
*   **(Placeholder for Video/Demo)**
*   *What to observe:*
    *   Synchronized leg movement.
    *   Storing energy in the landing (elasticity).

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

