# 3 DESIGN OF AI METROPOLIS

Motivated by observations described in [§2.2,](#page-2-0) we design AI Metropolis, an optimized simulation engine that serves as middleware between the developer-defined world and agents and the LLM serving engine, efficiently managing state updates and scheduling LLM queries. By allowing agents to progress at varying speeds based on their LLM call loads, AI Metropolis eliminates the need for frequent global synchronization, reducing false dependencies and maximizing parallelism. Algorithm 3 provides an overview of the new scheduling workflow adopted by AI Metropolis, contrasting it with the traditional time step synchronized scheduling shown in Algorithm [1.](#page-1-0)

