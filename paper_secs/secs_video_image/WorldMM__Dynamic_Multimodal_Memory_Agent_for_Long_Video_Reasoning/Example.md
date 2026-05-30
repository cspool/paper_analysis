# Example:

Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.

```
{ "named entities":
     ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
{ "triples": [
     ["Radio City", "located in", "India"],
     ["Radio City", "is", "private FM radio station"],
     ["Radio City", "started on", "3 July 2001"],
     ["Radio City", "plays songs in", "Hindi"],
     ["Radio City", "plays songs in", "English"],
     ["Radio City", "forayed into", "New Media"],
     ["Radio City", "launched", "PlanetRadiocity.com"],
     ["PlanetRadiocity.com", "launched in", "May 2008"],
     ["PlanetRadiocity.com", "is", "music portal"],
     ["PlanetRadiocity.com", "offers", "news"],
     ["PlanetRadiocity.com", "offers", "videos"],
     ["PlanetRadiocity.com", "offers", "songs"]
]}
```

Figure 11. Prompt for episodic triplet extraction.

<span id="page-23-0"></span>As an Event Summary Documentation Specialist, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

### # Specific Requirements

#### 1. Structure the Events Clearly

- Merge related events: Consolidate similar content into major events and arrange them in chronological order to ensure a smooth logical flow.
- Logical segmentation: Events can be grouped based on location, task, or theme. Each event should have a clear starting point, progression, and key turning points without any jumps or fragmentation in the information.

### 2. Retain Key Information

- All subjects' decisions and actions must be fully presented, including all critical first-person activities. Transitions between different parts, such as moving between floors or starting/ending a task, should be seamless.
- Any discussions, decisions, and task execution involving the primary character and other key individuals that impact the main storyline must be reflected. This includes recording, planning, and confirming matters, but in a concise manner.
- The purpose and method of key actions must be recorded, such as "ordering takeout using a phone" or "documenting a plan on a whiteboard."

