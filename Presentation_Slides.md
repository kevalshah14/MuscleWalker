# Bio-Inspired Robotics: MuscleWalker Project
## 25-Minute Presentation Outline

**Total Duration:** ~25 Minutes
**Target Slides:** ~12-15 Slides (~1.5 - 2 mins per slide)

---

### Slide 1: Title Slide
*   **Title:** MuscleWalker: Bio-Inspired Control of a Musculoskeletal Biped
*   **Subtitle:** Bio-Inspired Robotics Final Project
*   **Name:** [Your Name]
*   **Date:** [Date]
*   **Visual:** A high-quality render or screenshot of the robot from the MuJoCo viewer.

---

### Slide 2: Introduction - The Bio-Inspiration Process
*   **The Problem:**
    *   Traditional robots use rigid motors and gearboxes (stiff, jerky, inefficient).
    *   Biological motion is fluid, compliant, and energy-efficient due to elastic energy storage.
    *   *Challenge:* Controlling compliant, redundant muscle actuators is much harder than controlling servo motors.
*   **Goal:** Create a bipedal simulation that moves using simulated muscles and tendons, not direct torque.

### Slide 3: Model Organism & Design Inspirations
*   **Model Organism:** Bipedal mammals (Humans/Kangaroos).
*   **Key Inspirations:**
    *   **Musculoskeletal Architecture:** Bone structure driven by pulling forces only (muscles cannot push).
    *   **Passive Elasticity:** Using tendons to store energy during impact (like the Achilles tendon).
    *   **Agonist-Antagonist Pairs:** Flexor and Extensor muscles working in opposition to control joint stiffness and position.

### Slide 4: State of the Art (Quick Review)
*   **Classical Control:** Linear Inverted Pendulum (LIPM) + ZMP. Good for rigid robots (ASIMO), fails with soft tissues.
*   **Trajectory Optimization:** accurate but computationally heavy (hard for real-time compliance).
*   **Deep Reinforcement Learning (DRL):**
    *   *DeepMind (2017):* Showed emergent locomotion in stick figures.
    *   *MyoSuite/MyoLeg:* Recent benchmarks for musculoskeletal control.
    *   *Gap:* We are applying DRL to a custom, simplified muscle-walker to understand the fundamentals of muscle coordination.

---

### Slide 5: Methods - Simulation Platform
*   **Engine:** MuJoCo (Multi-Joint dynamics with Contact).
*   **Why MuJoCo?**
    *   Industry standard for contact physics.
    *   Built-in support for complex muscle/tendon actuators (Hill-type muscle models).
    *   Fast simulation speed allows for millions of training steps.
*   **Visual:** Screenshot of the XML model hierarchy or the MuJoCo GUI.

### Slide 6: Methods - Robot Design & Actuation
*   **Morphology:** 7-link biped (Torso, 2x Thigh, 2x Shin, 2x Foot).
*   **Actuation (The "Bio" Part):**
    *   **Spatial Tendons:** Routing paths defined in XML to mimic real muscle insertion points.
    *   **Active vs. Passive:**
        *   *Extensors (Anti-gravity):* High passive stiffness (spring-like) to support weight.
        *   *Flexors:* Active control for lifting legs.
*   **Visual:** Diagram of the leg showing where "muscles" attach (e.g., Hip Flexor vs. Hip Extensor).

### Slide 7: Methods - Control Strategy 1: Reinforcement Learning
*   **Algorithm:** Proximal Policy Optimization (PPO).
*   **Input (Observations):** Joint angles, velocities, and center-of-mass height.
*   **Output (Actions):** 6 continuous muscle activation signals [0, 1].
*   **Learning Process:** The agent explores random muscle twitches -> gets rewarded -> reinforces good actions.

### Slide 8: Methods - The Reward Function (Crucial)
*   **The Challenge:** The "lazy agent" problem (robot prefers falling or standing still to avoid negative rewards).
*   **Solution:** Multiplicative Reward Structure.
    *   `Reward = (Upright * Standing * Posture) * Velocity`
*   **Breakdown:**
    *   *Upright:* Torso must be vertical.
    *   *Standing:* Head must be at a certain height.
    *   *Velocity:* Must move forward.
    *   *Result:* If it falls OR stops moving, Reward = 0.

### Slide 9: Methods - Control Strategy 2: Model Predictive Control (MPPI)
*   **Approach:** Model-Based Control (Physics-aware).
*   **How it works:**
    1.  Simulate 100s of random muscle sequences in parallel.
    2.  Pick the sequences that don't fall over.
    3.  Execute the first step of the best average sequence.
*   **Comparison Goal:** Contrast "Instinctive" learning (RL) vs. "Planned" control (MPPI).

---

### Slide 10: Results - Live Demo / Video
*   **Action:** Switch to video or live simulation.
*   **Showcase:**
    1.  Untrained Agent (collapsing immediately).
    2.  MPPI Agent (struggling to balance, vibrating).
    3.  Final PPO Agent (walking gait).
*   *Note:* Highlight the compliance—how the knees bend slightly on impact (natural shock absorption).

### Slide 11: Results - Experimental Findings
*   **Training Curves:**
    *   Show TensorBoard graph: "Mean Reward" vs. "Timesteps".
    *   *Phase 1 (0-1M steps):* Learning to stand up.
    *   *Phase 2 (1M-3M steps):* Learning to shuffle forward.
    *   *Phase 3 (3M+ steps):* Stabilizing the gait.
*   **Gait Analysis:** The robot discovered a "bounding" gait (or walking gait), utilizing the passive stiffness of the extensor tendons.

### Slide 12: Results - Discussion & Comparison
*   **RL vs. MPPI:**
    *   *RL:* Smoother, more robust, "learned muscle memory." Hard to interpret (black box).
    *   *MPPI:* Good for immediate balance, but computationally expensive (laggy) and struggles with long-term horizons (walking).
*   **Do the results make sense?**
    *   Yes, the learned gait minimizes energy by using the passive tendons for support, aligning with biological energy conservation.

### Slide 13: Discussion - Challenges
*   **Tuning Muscle Stiffness:**
    *   *Too stiff:* Robot acts like a rigid servo robot (bouncy).
    *   *Too soft:* Robot collapses under its own weight.
*   **Local Optima:** The robot often learned to dive forward (faceplant) because it technically increased forward velocity for a split second before the episode ended.
    *   *Fix:* Strict early termination if torso height drops.

---

### Slide 14: Conclusions & Future Directions
*   **Summary:** Successfully created a bio-inspired walker that learns to coordinate muscles without knowing the equations of motion.
*   **Future Work:**
    *   **Metabolic Cost:** Add energy penalty to reward to force efficient walking.
    *   **3D Terrain:** Train on uneven ground.
    *   **Refined Anatomy:** Add bi-articular muscles (muscles crossing two joints) for better energy transfer, like the hamstrings.

### Slide 15: Questions?
*   Thank you!

