# The give-up circuit subgraph

**Source.** MaleCNS v1.0 (Berg et al., *Cell* 2026; CC-BY 4.0), the complete male
*Drosophila* CNS. We read the public flat-connectome tables from
`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`:

* `body-annotations-male-cns-v1.0-minconf-0.5.feather` (14.5 MB)
* `body-neurotransmitters-male-cns-v1.0.feather` (43.3 MB)
* `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` (508 MB)

The SHA-256 of every file is recorded in
`data/connectome/give_up_circuit.provenance.json`.

**Build.** `python scripts/build_connectome.py --download`, code in `src/connectome/`,
parameters in the `connectome:` section of `configs/base.yaml`. The output is
`data/connectome/give_up_circuit.npz`: node ids, edge list, synapse counts, node roles
and signs. Human-readable node and edge tables are written next to it
(`*_nodes.csv`, `*_edges.csv`), along with the provenance record (dataset, query,
resolved identifiers, counts per stage, date, raw-file and subgraph checksums, code
commit). Every training run re-verifies the subgraph checksum against the provenance
record.

**Hard rule.** A missing raw file, a failed download, a size mismatch, a missing cache
or a checksum mismatch raises `ConnectomeDataError` and stops the run. There is no code
path that builds a synthetic or small-world stand-in (`tests/test_connectome.py::
test_missing_cache_fails_loudly`, `test_corrupted_cache_fails_checksum`).

---

## 1. Resolving the cell types

The paper names the populations functionally. None of the literal strings in the
brief ("PPL1", "MBON-18", "MBON-19", "Tk-GAL4-FruM") is itself a MaleCNS `type`, so
each was resolved as follows.

| Population in the paper | How it was resolved | MaleCNS `type` | Neurons (bodyId) |
|---|---|---|---|
| PPL1 dopaminergic neurons | The PPL1 cluster is typed `PPL101`–`PPL108` in MaleCNS; these match hemibrain/FlyWire names. The regex `^PPL1[0-9]+$` matches exactly those 8 types and nothing else. | PPL101…PPL108 | 16: PPL101 (11327, 11900), PPL102 (11618, 13428), PPL103 (11752, 14182), PPL104 (21145, 515048), PPL105 (14225, 512326), PPL106 (529512, 529516), PPL107 (14705, 519616), PPL108 (12902, 15310) |
| MBON-18, MBON-19 (V2 cluster) | MaleCNS writes MBON numbers without a hyphen. `instance` confirms the compartments: `MBON18(a2sc)`, `MBON19(a2p3p)`. The paper finds that silencing MBON15–17 (R24H08) did *not* block defeat and concludes MBON-18/-19 are the relevant V2 types, so MBON17 and MBON17-like are **excluded**. | MBON18, MBON19 | 6: MBON18 (14221, 512788), MBON19 (26853, 33617, 36094, 47551) |
| Tk-GAL4^FruM aggression neurons | Resolved from the paper itself: "AVLP727m, a cell type that is annotated as Tk-GAL4FruM neuron in MCNS v0.9". The type is unchanged in v1.0. | AVLP727m | 5: 18792, 18987, 50960, 530190, 533662 |

All 27 core neurons have status `Traced`. MBON18/19 and AVLP727m are predicted
cholinergic, and all 16 PPL1 neurons dopaminergic, which is consistent with their
known identities.

## 2. The exact circuit (81 neurons)

The paper reports that at a threshold of **5 synapses** no neuron within 2 synapses
downstream of the V2 MBONs is AVLP727m. It traces MBON→AVLP727m connections as
**4-neuron paths** (MBON → X → Y → AVLP727m). We follow both choices:

* **Connecting neurons**: every neuron on a directed path of at most 3 synapses from
  {MBON18, MBON19} to AVLP727m, with every edge having at least 5 synapses. This gives
  **54 neurons**. There is no direct MBON→AVLP727m edge and no 2-synapse path, which
  reproduces the paper's negative result on v1.0. Their types are listed in
  `give_up_circuit_nodes.csv` (`stage == connecting`). They include SMP106 (×5), SMP108,
  CRE021, CRE083, LHCENT9, SIP100m, AVLP742m/758m/732m/733m, OA-VPM4 (octopaminergic),
  5-HTPMPD01 (serotonergic) and the descending neuron DNp62.
* **PPL1 → V2 MBON** is monosynaptic in v1.0: PPL105 → MBON18 has 434 synapses and
  PPL105 → MBON19 has 98. PPL104 and PPL106 contacts fall below 5 synapses. No
  intermediate neurons are needed on this leg.

Core plus connecting neurons gives **81 neurons**, below the 100–500 target.

## 3. One-hop expansion (+177 → 258 neurons)

**The exact circuit is smaller than 100 neurons, so it was expanded by one synaptic
hop.** A full hop (every neuron with any ≥5-synapse contact) adds about 6,400 neurons,
so the hop is restricted by connection strength. We add every neuron that exchanges at
least **400 synapses** (inputs plus outputs, over edges of at least 5 synapses) with the
81-neuron circuit. This adds **177 neurons**. They are mostly central-brain intrinsic
neurons (166), plus 5 descending, 4 ascending and 2 visual projection neurons. Among them
are 16 further MBONs (MBON01/02/04/11/13/30–35), APL, DPM, PPL201 and many AVLP/SMP
types.

The value 400 was chosen to land in the middle of the range. The alternatives give
250 → 500 neurons, 300 → 386, 500 → 197. The size of the network also sets per-step
compute. The results have **not** been tested for sensitivity to this choice (see
RESULTS.md, *What we could not establish*).

**Known consequence.** Kenyon cells are the main input to MBON18/19 (15,336 of their
input synapses), but no single KC exchanges more than about 21 synapses with the
circuit. A per-neuron strength criterion therefore excludes all of them. The subgraph
represents the MBON *output* side of the mushroom body, not its KC input layer.

## 4. The subgraph

The induced subgraph on the 258 neurons, keeping edges with at least 5 synapses:

| | |
|---|---|
| neurons | 258 (27 core, 54 connecting, 177 expansion) |
| directed edges | 4,952, carrying 220,286 synapses; density 7.4% |
| self-loops dropped | 1 |
| reciprocity | 0.43, i.e. 43% of edges have a reverse partner |
| average clustering (undirected) | 0.39 |
| input → output shortest path | 2 synapses minimum, median 3; all 16 × 5 pairs reachable |
| subgraph SHA-256 | `48ef287d…64902ed` (full value in the provenance file) |

**Roles, which are the same in every circuit arm.** Context input enters through a
dense projection onto the **16 PPL1 neurons**, the dopaminergic teaching signal. The
policy is read out linearly from the **5 AVLP727m neurons**, the aggression neurons
whose inhibition implements the give-up decision. All other neurons are hidden.

**Signs (Dale's law).** Each neuron's MaleCNS consensus transmitter prediction sets
the sign of all its outgoing edges: acetylcholine +1 (137 neurons); GABA, glutamate
and histamine −1 (95; all act on chloride channels in *Drosophila*); dopamine,
octopamine, serotonin or unclear 0 (26; modulatory or unknown, left unconstrained).
These are predictions from EM images, not measurements. In the sign-constrained
condition a +1 edge is parameterised as `softplus(w) ≥ 0` and a −1 edge as
`−softplus(w) ≤ 0`.

## 5. Controls

Both controls keep the node set, the node roles and the node signs. Edge signs follow
the presynaptic neuron, so each control has exactly the same trainable parameter
count as the connectome arm. Each control graph is drawn fresh from the run seed, so
the controls average over many wirings rather than one.

| | connectome | degree-preserving rewire | random sparse |
|---|---|---|---|
| edges | 4,952 | 4,952 | 4,952 |
| in/out degree of every neuron | real | **identical to real** | not preserved (max 31 vs 59) |
| reciprocity | 0.43 | ≈ 0.10 | ≈ 0.08 |
| clustering | 0.39 | ≈ 0.21 | ≈ 0.14 |
| overlap with real edge set | 1 | ≈ 0.12 | ≈ 0.07 |
| median input→output path | 3 | 2 | 2 |

The rewire runs 10 × E directed double-edge swaps `(a→b, c→d) → (a→d, c→b)`, rejecting
any swap that would create a self-loop or a duplicate edge.
`test_degree_preserving_rewire_preserves_degrees` checks that every neuron's in- and
out-degree is unchanged.

So the rewired control destroys the circuit's specific wiring, including its strong
reciprocity and clustering, but keeps sparsity and degree sequence. The random
control also destroys the degree sequence. The contrasts are:

* connectome vs rewired: *specific wiring*
* rewired vs random: *degree structure*
* random vs dense: *sparsity and the recurrent-circuit architecture*

---

## 6. Why not connectorch

`connectorch` (github.com/us/connectorch, MIT) was evaluated by reading its source at
commit `cd7f809` (2026-09-18):

* It is **not on PyPI**. `pip install connectorch` fails ("No matching distribution"),
  despite the README.
* Its MaleCNS loader (`datasets/malecns.py`) reads the same public flat-connectome
  feather files we use, and supports metadata filtering (`Connectome.where(cell_type=…)`
  plus `subgraph()`). Neuron-level filtering by cell type *is* possible. There is no
  path-based or strength-thresholded partner selection, which is what we need.
* Its model (`nn.ConnectomeRNN`) is a recurrent network over the whole graph, and its
  controls (`transforms/rewire.py`) include degree-preserving and random-topology
  rewires. The design is sound, but it compiles a larger model than we want, with its
  own dynamics and parameterisation.

We therefore wrote a small loader against the same raw files, about 250 lines in
`src/connectome/`, adopting its conventions:

* the `traced-only` connectivity variant;
* the transmitter-to-sign mapping (ACh +, GABA/Glu −, modulators unsigned);
* refusing to download or substitute data without being told.

Our own sparse layer (`SparseCircuitPolicy`) stores only the edge weights as parameters,
so the no-new-connections rule holds by construction
(`test_masked_weights_cannot_grow_new_connections`).
