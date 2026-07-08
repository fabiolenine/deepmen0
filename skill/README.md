# DeepMem0 memory skill

`deepmem0-memory.skill` is a Claude **Skill** that governs *when* and *how* Claude
uses the DeepMem0 MCP tools (`search_memories`, `add_memory`, `update_memory`) to
keep durable memory across conversations — retrieving context at the start of a
technical chat and saving only facts worth keeping, under a strict save/discard
filter so retrieval returns signal instead of noise.

The `.skill` file is a bundle (a zip of `deepmem0-memory/SKILL.md`); unzip it to
read or edit the rubric:

```bash
unzip -p deepmem0-memory.skill deepmem0-memory/SKILL.md   # print
unzip deepmem0-memory.skill                               # extract to ./deepmem0-memory/
```

## Prerequisite: connect the DeepMem0 MCP server

The skill only pays off if Claude can actually reach the DeepMem0 tools. Connect
the DeepMem0 MCP server first — as a **connector** in Claude Desktop, or an MCP
server entry in Claude Code — so `search_memories` / `add_memory` are available.

## Two ways to trigger it — pick by where you live

Memory only works if something makes Claude *reach for it* on every relevant turn.
That continuous trigger looks different depending on the surface.

### Claude Code → hooks + skill

Claude Code discovers skills and can fire them from its **hook** system, and the
skill's own description (`USE SEMPRE que a conversa for técnica…`) is enough to
get it loaded when the conversation turns technical. Install the bundle and let
the hook/trigger machinery do the continuous part.

### Claude Desktop (chat) → always-on instruction + skill

Claude Desktop has **no hook system** like Claude Code, so the continuous trigger
has to be a short, always-on instruction. Put a lean rubric in your **custom
instructions** (or a **Project's** instructions):

```text
At the start of technical conversations, use search_memories (DeepMem0) to
retrieve relevant context. During the conversation, when a durable fact emerges
— an architecture decision, a resolved config, a stable preference, a
hardware/infra detail — save it with add_memory without asking for confirmation.
Do not save: ephemeral questions, intermediate reasoning, throwaway code.
```

Then leave the detail — memory format, categorization, dedup, security, examples
— to the **Skill**, keeping the instruction light. That is **progressive
disclosure**: the instruction fires, and the skill supplies the *how* only when a
technical conversation actually calls for it.

## Install the skill in Claude Desktop

1. Settings → **Capabilities → Skills**.
2. Upload / import `deepmem0-memory.skill`.
3. Add the always-on rubric above to your custom (or Project) instructions.

## Why split instruction and skill

- An **always-on long instruction** would burn context on every message and drift
  over time.
- A **skill with no trigger** never fires in Desktop, because there is no hook to
  invoke it.
- **Light trigger + deep skill** gives you the best of both: a few always-in-context
  lines get Claude to *reach for memory*, and the full rubric (the decision test,
  what to save vs. discard, dedup, the never-save security rule, worked examples)
  loads only when it is actually needed.

---

## Português

`deepmem0-memory.skill` é uma **Skill** do Claude que governa *quando* e *como* o
Claude usa as ferramentas MCP do DeepMem0 (`search_memories`, `add_memory`,
`update_memory`) para manter memória durável entre conversas — recuperando
contexto no início de uma conversa técnica e salvando só o que merece ser retido,
com um filtro rigoroso do que salvar e do que descartar, para que a recuperação
traga sinal em vez de ruído.

O arquivo `.skill` é um pacote (um zip de `deepmem0-memory/SKILL.md`); descompacte
para ler ou editar a rubrica (`unzip -p deepmem0-memory.skill deepmem0-memory/SKILL.md`).

### Pré-requisito: conectar o servidor MCP do DeepMem0

A skill só compensa se o Claude conseguir alcançar as ferramentas do DeepMem0.
Conecte o servidor MCP do DeepMem0 primeiro — como **connector** no Claude Desktop,
ou como entrada de servidor MCP no Claude Code.

### Dois jeitos de disparar — escolha pelo lugar onde você vive

**Se você vive no Claude Desktop (chat) → instrução permanente + Skill.** O Desktop
não tem o mesmo sistema de *hooks* do Claude Code, então o gatilho contínuo tem que
ser uma instrução curta sempre-ativa. Coloque nas **custom instructions** (ou nas
instruções de um **Projeto**) uma rubrica enxuta:

```text
No início de conversas técnicas, use search_memories (DeepMem0) para recuperar
contexto relevante. Ao longo da conversa, quando surgir um fato durável — decisão
de arquitetura, config resolvida, preferência estável, dado de hardware/infra —
salve com add_memory sem pedir confirmação. Não salve: perguntas efêmeras,
raciocínio intermediário, código descartável.
```

E deixe o detalhamento (formato da memória, categorização, dedup, segurança,
exemplos) na **Skill**, para manter a instrução leve. Isso é **progressive
disclosure**: a instrução dispara, a skill fornece o *como* só quando necessário.

**Se você vive no Claude Code → hooks + Skill.** O Claude Code descobre skills e
pode dispará-las pelo sistema de *hooks*; a própria descrição da skill já basta
para carregá-la quando a conversa fica técnica. Instale o pacote e deixe o
mecanismo de gatilho cuidar da parte contínua.

### Instalar a skill no Claude Desktop

1. Configurações → **Capabilities → Skills**.
2. Faça upload / importe `deepmem0-memory.skill`.
3. Adicione a rubrica sempre-ativa acima às suas instruções (custom ou de Projeto).

### Por que separar instrução e skill

Uma instrução longa sempre-ativa gastaria contexto a cada mensagem e se desviaria
com o tempo; uma skill sem gatilho nunca dispara no Desktop. Gatilho leve + skill
profunda dá o melhor dos dois: poucas linhas sempre em contexto fazem o Claude
*buscar a memória*, e a rubrica completa (o teste de decisão, o que salvar vs.
descartar, dedup, a regra de nunca salvar segredos, exemplos) carrega só quando
realmente é preciso.
