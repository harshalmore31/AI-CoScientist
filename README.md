![AI-CoScientist](https://storage.googleapis.com/gweb-research2023-media/images/AICoScientist-1-Components.width-1250.png)

# AI-CoScientist

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/swarms-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

A multi-agent AI framework for collaborative scientific research, implementing the "Towards an AI Co-Scientist" methodology with tournament-based hypothesis evolution, peer review systems, and intelligent agent orchestration.

## Features

🧠 **Multi-Agent Architecture**: Specialized agents for hypothesis generation, peer review, ranking, evolution, and meta-analysis  
🏆 **Tournament-Based Selection**: Elo rating system for hypothesis ranking through pairwise comparisons  
📊 **Comprehensive Review System**: Scientific soundness, novelty, testability, and impact assessment  
🔄 **Iterative Refinement**: Meta-review guided evolution with strategic hypothesis improvement  
🎯 **Diversity Control**: Proximity analysis to maintain hypothesis diversity and reduce redundancy  
📈 **Execution Metrics**: Detailed performance tracking and agent timing analytics  
💾 **State Persistence**: Save and resume research workflows with agent state management  
🛡️ **Robust Error Handling**: Graceful fallbacks and recovery mechanisms for production reliability

## Installation

You can install the package using pip:

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install swarms loguru python-dotenv
```

## Quick Start

```python
from ai_coscientist import AIScientistFramework

# Initialize the AI Co-scientist Framework
ai_coscientist = AIScientistFramework(
    model_name="gpt-4",
    max_iterations=3,
    hypotheses_per_generation=10,
    tournament_size=8,
    evolution_top_k=3,
    verbose=True
)

# Define your research goal
research_goal = "Develop novel approaches for improving reasoning capabilities in large language models"

# Run the research workflow
results = ai_coscientist.run_research_workflow(research_goal)

# Access the results
print(f"Generated {len(results['top_ranked_hypotheses'])} top hypotheses")
for i, hypothesis in enumerate(results['top_ranked_hypotheses'], 1):
    print(f"{i}. {hypothesis['text']}")
    print(f"   Elo Rating: {hypothesis['elo_rating']}")
    print(f"   Win Rate: {hypothesis['win_rate']}%")
```

## Architecture

The AI-CoScientist framework consists of 8 specialized agents:

- **Generation Agent**: Creates novel research hypotheses
- **Reflection Agent**: Peer review and scientific critique
- **Ranking Agent**: Hypothesis ranking and selection
- **Evolution Agent**: Hypothesis refinement and improvement
- **Meta-Review Agent**: Cross-hypothesis insight synthesis
- **Proximity Agent**: Similarity analysis and diversity control
- **Tournament Agent**: Pairwise hypothesis comparison
- **Supervisor Agent**: Workflow orchestration and planning

## Advanced Usage

### Custom Configuration

```python
ai_coscientist = AIScientistFramework(
    model_name="claude-3-sonnet",
    max_iterations=5,
    base_path="./custom_states",
    verbose=True,
    tournament_size=12,
    hypotheses_per_generation=15,
    evolution_top_k=5,
)
```

### State Management

```python
# Save agent states
ai_coscientist.save_state()

# Load previous states
ai_coscientist.load_state()
```

### Results Analysis

```python
results = ai_coscientist.run_research_workflow(research_goal)

# Execution metrics
metrics = results['execution_metrics']
print(f"Total time: {results['total_workflow_time']:.2f}s")
print(f"Hypotheses generated: {metrics['hypothesis_count']}")
print(f"Reviews completed: {metrics['reviews_count']}")
print(f"Tournament rounds: {metrics['tournaments_count']}")

# Meta-review insights
insights = results['meta_review_insights']
print("Strategic recommendations:", insights.get('strategic_recommendations'))
```

## Code Quality 🧹

- `make style` to format the code
- `make check_code_quality` to check code quality (PEP8 basically)
- `black .`
- `ruff . --fix`


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Documentation

For detailed documentation, see [DOCS.md](DOCS.md).

## Citation

If you use AI-CoScientist in your research, please cite:

```bibtex
@software{ai_coscientist,
  title={AI-CoScientist: A Multi-Agent Framework for Collaborative Scientific Research},
  author={The Swarm Corporation},
  year={2024},
  url={https://github.com/The-Swarm-Corporation/AI-CoScientist}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: kye@swarms.world
- 💬 Discord: [Join our community](https://discord.gg/swarms-999382051935506503)
- 🐦 Twitter: [@kyegomezb](https://x.com/kyegomezb)
