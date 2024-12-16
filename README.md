# Arxiv-to-Podcast 🎤 with Naptha & LangGraph

Transform academic research papers into engaging podcast conversations using LangGraph's workflows and with ease Naptha Node deployment.

## Inspiration and Reference

This project is inspired by and builds upon the [Paper-to-Podcast](https://github.com/Azzedde/paper_to_podcast) project

## Overview

Arxiv-to-Podcast creates natural, multi-persona discussions from academic papers using:
- State-managed conversation flow
- Hierarchical document processing
- OpenAI's text-to-speech for voice synthesis

### Architecture

The system uses LangGraph to orchestrate the following workflow:
1. **PDF Processing Node**: Parses academic papers with section hierarchy preservation
2. **Planning Node**: Generates a structured conversation outline
3. **Discussion Node**: Creates dynamic dialogue between personas
4. **Enhancement Node**: Refines and polishes the conversation
5. **Audio Generation Node**: Converts text to natural speech podcast conversation

### Key Features

- **Structured Conversation Flow**: Uses LangGraph's state management to maintain context and conversation history
- **Section-Aware Processing**: Maintains paper structure through custom Section dataclass
- **RAG Integration**: Leverages document chunks for accurate content generation
- **Multi-Persona Dialogue**: Three distinct voices (Host, Expert, Learner)
- **Cost Efficient**: ~$0.16 for a 9-minute podcast from a 19-page paper
