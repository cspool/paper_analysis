# <span id="page-7-4"></span>*A. Interrupt Delivery Strategy*

To answer Q1, we investigate three types of SPIs, including network interrupts, mouse interrupts, and keyboard interrupts. As the first step of our study, we validate whether they can be delivered to the attack core using default scheduling strategy. To this end, we separately request the three types of resources (i.e, mouse movement, keystroke, and network package) from a sender process, thereby triggering the corresponding SPIs. At the same time, we use TIDE to detect interrupts from another receiver process. Our goal is to validate whether an attack process can detect the SPIs triggered by the victim, or only on some specific cores as previous attacks on x86 [\[7\]](#page-13-6), [\[25\]](#page-13-24), [\[18\]](#page-13-17) or Arm [\[23\]](#page-13-22) architectures.

Experimental setup. To generate network interrupts, we use udp protocol to send a packet of 1 B or 9 KB to one of our controlled ports. To generate mouse interrupts, we use a C-based program that invokes the CGEventRef API to simulate user input, thereby moving the mouse in place. To trigger keyboard interrupts, we use another C-based program to repeatedly press backspace. To avoid disrupting normal system behavior, we set the interrupt request frequency to 200 kHz for network interrupts, and 5 kHz for both mouse and keyboard interrupts.

First, we run our experiments for 10 seconds to evaluate the *efficiency* under default macOS settings and just let the macOS

<span id="page-7-3"></span>TABLE II: Efficiency for three types of interrupts on our tested machines. Mac mini devices are not equipped with a mouse or keyboard, and are therefore marked as '–'.

| Setting               |        | Network (1 B) Network (9 KB) Mouse Keyboard |      |      |
|-----------------------|--------|---------------------------------------------|------|------|
| MacBook Pro 2021      | 8.01   | 7.41                                        | 1.13 | 4.32 |
| Mac mini 2023         | 33.74  | 10.96                                       | –    | –    |
| MacBook Air 2023      | 122.82 | 111.2                                       | 1.06 | 0.56 |
| MacBook Pro 2023      | 149.27 | 180.70                                      | 0.76 | 2.91 |
| Mac mini 2020 (Cloud) | 6.03   | 6.05                                        | –    | –    |
| Mac mini 2024 (Cloud) | 12.47  | 12.43                                       | –    | –    |

schedule them. Second, we examine the influence of the CPU core on which our program executes. As macOS does not provide the interface to control the affinity from user-space, we employ CoreBinder[4](#page-7-2) , a kernel extension that allows pinning threads to specific CPU cores on Apple silicon [\[73\]](#page-14-29). With CoreBinder, we sequentially assign the sender process to each core while ensuring that the receiver process runs on a different core on Mac mini 2023. Since this machine has 8 cores, we evaluate the *efficiency* of network interrupts using 9 kB under 56 settings. It is important to note that while both installing CoreBinder and cntvct\_el0 fall out of our threat model, they are used solely for validation and are not necessary in our attacks in Section [V.](#page-8-0)

Experimental results. Table [II](#page-7-3) presents the results on all our tested machines under default system setting. Since the underlying delivery algorithm is closed source, we cannot explain the variations in efficiency across settings. Nonetheless, we observe that these SPIs can be detected under all the settings although their efficiencies vary. Besides, as our tested Mac mini 2023 has 8 cores, there are 56 different possibilities for our sender core and receiver core. Figure [5](#page-8-1) shows the efficiencies across these 56 settings. When the two processes are pinned to different cores, we obtain stable efficiencies, which is 10.73 ± 0.20 (avg ± std). The results indicate that interrupts are delivered uniformly across all cores and irrespective of the sender's core. We note that this mechanism is completely different from that observed in Linux-based systems [\[7\]](#page-13-6), where device interrupts are routed to a fixed core via interrupt affinity. In addition, we also observe that the total number of interrupts received does not exactly match the rate at which we trigger SPIs. We hypothesize that macOS aggregates multiple resource requests into a single interrupt to reduce context-switch overhead.

Observation 1: Unlike Linux-based systems, the combination of Apple silicon and macOS does not adopt an interrupt affinity mechanism, and deliver shared peripheral interrupts to a fixed core.

