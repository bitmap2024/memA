Chat Bot Memory
├── Profile Memory
│   ├── fact
│   ├── preference
│   ├── relationship
│   ├── goal
│   ├── constraint
│   ├── skill
│   ├── portrait
│   └── communication_style
│
├── Episodic Memory
│   ├── conversation_event
│   ├── life_event
│   ├── project_event
│   ├── relationship_event
│   ├── decision_event
│   └── unresolved_event
│
├── State Memory
│   ├── current_focus
│   ├── recent_mood
│   ├── emotional_need
│   ├── relationship_state
│   ├── task_state
│   └── short_term_context
│
└── Core Memory (蒸馏层，始终注入 system prompt)
    ├── identity              ← Profile: fact, relationship
    ├── interaction_guide     ← Profile: preference, constraint, communication_style, portrait
    ├── current_focus         ← State: current_focus, task_state + Episodic: project_event, unresolved_event
    ├── emotional_state       ← State: recent_mood, emotional_need + Profile: portrait
    └── pending_followups     ← Episodic: unresolved_event, decision_event + State: task_state, relationship_state