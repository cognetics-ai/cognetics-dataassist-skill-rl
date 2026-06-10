"""SkillSQL-RL workflow layer.

Two workflow paths exist and serve different purposes:

INFERENCE (production, single best answer per question):
    app.agents.text2sql_workflow.agent.build_agent()
    SequentialAgent: Directory -> ContextBuilder[+SkillBank] -> Generator ->
      SqlRefinementLoop[Critic -> Refiner -> LoopExit] -> Validator[+Reward] ->
      Optimizer -> SkillDistillation

TRAINING (GRPO, multi-candidate rollouts, benchmark):
    skillsql.workflow.text2sql_workflow.build_root_workflow()  (this package)
    @node workflow: retrieve_schema -> generate*G -> verify_and_select

Both paths share:
    skillsql.verification.*    formal verifier (static gates, obligations, reward)
    skillsql.catalog.*         pgvector semantic catalog
    skillsql.skillbank.*       SkillBank retrieval
    skillsql.connectors.*      abstract factory connector
"""
