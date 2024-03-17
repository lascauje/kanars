// Neural Network Architecture:
//                                                   Multi Layer
//                  ┌────────────────────────────────────────────────────────────────────┐
//                  │                                                                    │
//                  ▼                               Hidden Layer                         ▼
//                                    ┌──────────────┐       ┌───────────────┐
//                              ┌────►│              │       │               │
//                              │     │              │       │               │
//                 Input Layer  │     │    Neuron    │  ...  │    Neuron     │
//              ┌─────────────┐ │     │    Σ tanh    │       │    Σ tanh     │
//              │             ├─┼────►│              │       │               │
//      ┌──►    │             │ │     └──────────────┘       └───────────────┘
//      |       │   Neuron    ├─┼───┐                                                   Output Layer
//      |       │   Σ tanh    │ │   │ ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
//      |       │             ├─┼─┐ └►│               │       │               │       │               │
//     X Data   └─────────────┘ │ │   │               │       │               │       │               │
//      |                       │ │   │    Neuron     │  ...  │    Neuron     │  ...  │    Neuron     ├─────► Loss SE
//      |       ┌─────────────┐ │ │ ┌►│    Σ tanh     │       │    Σ tanh     │       │    Σ tanh     │
//      |       │             │ │ │ │ │               │       │               │       │               │
//      └──►    │             ├─┘ │ │ └───────────────┘       └───────────────┘       └───────────────┘
//              │   Neuron    │   │ │
//              │   Σ tanh    ├───┼─┘ ┌───────────────┐       ┌───────────────┐
//              │             │   │   │               │       │               │
//              └─────────────┴─┐ └──►│               │  ...  │               │
//                              │     │    Neuron     │       │    Neuron     │
//                              └────►│    Σ tanh     │       │    Σ tanh     │
//                                    │               │       │               │
//                                    └───────────────┘       └───────────────┘

// Design:
// Activation function is tanh, Loss function is squared error
// Arena allocator pattern (references are used to handle tensors, and to prevent lifetime hacking)
// All operations and tensors are saved (introspective feature)
// No recursion, loop is better with huge data (no stackoverflow)
// Simple API (train, predict, etc.) and basic source code, but powerful enough for non-linear classification
// May not be a super Rust code, but it passes the clippy pedantic compilation!
// The entry point is NeuralNet struct

use std::fs::File;
use std::io::Write;
use std::vec;

use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub type TensorRef = usize;
pub type NeuronRef = usize;
pub type LayerRef = usize;
pub type MultilayerRef = usize;

#[derive(Clone, Copy)]
pub enum Operation {
    Add,
    Sub,
    Mult,
    Pow2,
    Tanh,
}

pub struct Tensor {
    label: String,
    value: Option<f64>,
    grad: Option<f64>,
    op: Option<Operation>,
    curr_tref: Option<TensorRef>,
    left_tref: Option<TensorRef>,
    right_tref: Option<TensorRef>,
}

impl Tensor {
    #[must_use]
    pub fn new(
        label: &str,
        value: Option<f64>,
        grad: Option<f64>,
        op: Option<Operation>,
        curr_tref: Option<TensorRef>,
        left_tref: Option<TensorRef>,
        right_tref: Option<TensorRef>,
    ) -> Self {
        Tensor {
            label: String::from(label),
            value,
            grad,
            op,
            curr_tref,
            left_tref,
            right_tref,
        }
    }
}

pub struct Neuron {
    nin: usize,
    weights_tref: Vec<TensorRef>,
    bias_tref: TensorRef,
    values_tref: Vec<TensorRef>,
}

impl Neuron {
    #[must_use]
    pub fn new(nin: usize, weights_tref: Vec<TensorRef>, bias_tref: TensorRef) -> Self {
        Neuron {
            nin,
            weights_tref,
            bias_tref,
            values_tref: Vec::new(),
        }
    }
}

pub struct Layer {
    nin: usize,
    nout: usize,
    neurons_nref: Vec<NeuronRef>,
    values_tref: Option<Vec<TensorRef>>,
}

impl Layer {
    #[must_use]
    pub fn new(
        nin: usize,
        nout: usize,
        neurons_nref: Vec<NeuronRef>,
        values_tref: Option<Vec<TensorRef>>,
    ) -> Self {
        Layer {
            nin,
            nout,
            neurons_nref,
            values_tref,
        }
    }
}

pub struct Multilayer {
    layers_lref: Vec<LayerRef>,
    value_tref: Option<TensorRef>,
}

impl Multilayer {
    #[must_use]
    pub fn new(layers_lref: Vec<LayerRef>, value_tref: Option<TensorRef>) -> Self {
        Multilayer {
            layers_lref,
            value_tref,
        }
    }
}

pub struct Engine {
    tensors: Vec<Tensor>,
    neurons: Vec<Neuron>,
    layers: Vec<Layer>,
    multilayers: Multilayer,
    xis_ref: Vec<Vec<TensorRef>>,
    xis_pred_ref: Vec<Vec<TensorRef>>,
    losses_ref: Vec<TensorRef>,
    topo: Vec<TensorRef>,
    topo_reverse: Vec<TensorRef>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    #[must_use]
    pub fn new() -> Self {
        Engine {
            tensors: Vec::new(),
            neurons: Vec::new(),
            layers: Vec::new(),
            multilayers: Multilayer {
                layers_lref: Vec::new(),
                value_tref: None,
            },
            xis_ref: Vec::new(),
            xis_pred_ref: Vec::new(),
            losses_ref: Vec::new(),
            topo: Vec::new(),
            topo_reverse: Vec::new(),
        }
    }

    pub fn tensor(&mut self, value: f64, label: &str) -> TensorRef {
        let idx = self.tensors.len();
        self.tensors.push(Tensor::new(
            label,
            Some(value),
            None,
            None,
            Some(idx),
            None,
            None,
        ));
        idx
    }

    pub fn tensor_value(&mut self, idx: TensorRef) -> f64 {
        self.tensors[idx].value.unwrap_or(0.0)
    }

    pub fn tensor_add(&mut self, l_idx: TensorRef, r_idx: TensorRef, label: &str) -> TensorRef {
        let idx = self.tensors.len();
        self.tensors.push(Tensor::new(
            label,
            None,
            None,
            Some(Operation::Add),
            Some(idx),
            Some(l_idx),
            Some(r_idx),
        ));
        idx
    }

    pub fn tensor_sub(&mut self, l_idx: TensorRef, r_idx: TensorRef, label: &str) -> TensorRef {
        let idx = self.tensors.len();
        self.tensors.push(Tensor::new(
            label,
            None,
            None,
            Some(Operation::Sub),
            Some(idx),
            Some(l_idx),
            Some(r_idx),
        ));
        idx
    }

    pub fn tensor_mult(&mut self, l_idx: TensorRef, r_idx: TensorRef, label: &str) -> TensorRef {
        let idx = self.tensors.len();
        self.tensors.push(Tensor::new(
            label,
            None,
            None,
            Some(Operation::Mult),
            Some(idx),
            Some(l_idx),
            Some(r_idx),
        ));
        idx
    }

    pub fn tensor_pow2(&mut self, l_idx: TensorRef, label: &str) -> TensorRef {
        let idx = self.tensors.len();
        self.tensors.push(Tensor::new(
            label,
            None,
            None,
            Some(Operation::Pow2),
            Some(idx),
            Some(l_idx),
            None,
        ));
        idx
    }

    pub fn tensor_tanh(&mut self, l_idx: TensorRef, label: &str) -> TensorRef {
        let idx = self.tensors.len();
        self.tensors.push(Tensor::new(
            label,
            None,
            None,
            Some(Operation::Tanh),
            Some(idx),
            Some(l_idx),
            None,
        ));
        idx
    }

    // Make clippy happy...
    #[allow(clippy::missing_panics_doc)]
    pub fn tensor_loss(&mut self, ytarget: &[TensorRef], ypred: &[TensorRef]) -> TensorRef {
        assert!(ytarget.len() == ypred.len());

        let mut vref_acc: TensorRef = 0;

        for i in 0..ytarget.len() {
            let vref_sub = self.tensor_sub(
                ytarget[i],
                ypred[i],
                format!("ytarget{i}-ypred{i}").as_str(),
            );
            self.losses_ref.push(vref_sub);

            let vref_curr =
                self.tensor_pow2(vref_sub, format!("pow2(ytarget{i}-ypred{i})").as_str());
            self.losses_ref.push(vref_curr);

            if i == 0 {
                vref_acc = vref_curr;
            } else {
                vref_acc = self.tensor_add(vref_acc, vref_curr, "Σ(pow2(ytargeti-ypredi))");
                self.losses_ref.push(vref_acc);
            }
        }

        vref_acc
    }

    pub fn neuron(&mut self, nin: usize) -> NeuronRef {
        let idx = self.neurons.len();
        let mut rng = StdRng::seed_from_u64(12345);

        let distr = Uniform::new(-1.0, 1.0);

        let weights: Vec<TensorRef> = (0..nin)
            .map(|v| self.tensor(rng.sample(distr), format!("w{v}").as_str()))
            .collect();
        let bias = self.tensor(0.0, "b1");

        self.neurons.push(Neuron::new(nin, weights, bias));
        idx
    }

    // Make clippy happy...
    #[allow(clippy::missing_panics_doc)]
    pub fn neuron_connect(&mut self, nref: NeuronRef, trefs: &[TensorRef]) -> TensorRef {
        assert!(trefs.len() == self.neurons[nref].nin);

        let len = trefs.len();
        let mut vref_acc = 0;

        for (i, tref) in trefs.iter().enumerate().take(len) {
            let vref_curr = self.tensor_mult(
                *tref,
                self.neurons[nref].weights_tref[i],
                format!("v{i}*w{i}").as_str(),
            );
            self.neurons[nref].values_tref.push(vref_curr);

            if i == 0 {
                vref_acc = vref_curr;
            } else {
                vref_acc = self.tensor_add(vref_acc, vref_curr, "Σ(vi*wi)");
                self.neurons[nref].values_tref.push(vref_acc);
            }
        }

        vref_acc = self.tensor_add(vref_acc, self.neurons[nref].bias_tref, "Σ(vi*wi)+b1");
        self.neurons[nref].values_tref.push(vref_acc);

        vref_acc = self.tensor_tanh(vref_acc, "tanh(Σ(vi*wi)+b1)");
        self.neurons[nref].values_tref.push(vref_acc);

        vref_acc
    }

    pub fn layer(&mut self, nin: usize, nout: usize) -> LayerRef {
        let idx = self.layers.len();
        let mut nrefs = vec![];

        for _ in 0..nout {
            let nref = self.neuron(nin);
            nrefs.push(nref);
        }

        self.layers.push(Layer::new(nin, nout, nrefs, None));
        idx
    }

    pub fn layer_connect(&mut self, lref: LayerRef, trefs: &[TensorRef]) -> Vec<TensorRef> {
        let mut v_tref = vec![];

        for nref in self.layers[lref].neurons_nref.clone() {
            let v = self.neuron_connect(nref, trefs);

            v_tref.push(v);
        }

        self.layers[lref].values_tref = Some(v_tref.clone());
        v_tref
    }

    pub fn multilayer(&mut self, nin_outs: &[usize]) {
        let mut lrefs = vec![];

        for i in 0..nin_outs.len() - 1 {
            let lref = self.layer(nin_outs[i], nin_outs[i + 1]);
            lrefs.push(lref);
        }

        self.multilayers.layers_lref = lrefs;
    }

    pub fn multilayer_connect(&mut self, xrefs: &[TensorRef]) -> TensorRef {
        let mut v_ref = xrefs.to_owned();

        for lref in self.multilayers.layers_lref.clone() {
            v_ref = self.layer_connect(lref, &v_ref);
        }

        self.multilayers.value_tref = Some(v_ref[0]);
        v_ref[0]
    }

    fn topo_sort_find_neigh(&self, root_idx: TensorRef) -> Vec<TensorRef> {
        let mut root_neighs: Vec<TensorRef> = vec![root_idx];

        if let Some(l_idx) = self.tensors[root_idx].left_tref {
            root_neighs.push(l_idx);
        }
        if let Some(r_idx) = self.tensors[root_idx].right_tref {
            root_neighs.push(r_idx);
        }
        root_neighs
    }

    fn topo_sort(&self) -> Vec<TensorRef> {
        let mut visited: Vec<bool> = vec![false; self.tensors.len()];
        let mut stack: Vec<TensorRef> = vec![];

        for idx in 0..self.tensors.len() {
            if !visited[idx] {
                let mut helper_stack: Vec<Vec<TensorRef>> = vec![self.topo_sort_find_neigh(idx)];

                while !helper_stack.is_empty() {
                    if let Some(curr) = helper_stack.pop() {
                        let root = curr[0];
                        visited[root] = true;

                        let mut found = false;
                        for (idx_n, vis) in visited.iter().enumerate().take(curr.len()).skip(1) {
                            if !vis {
                                helper_stack.push(vec![root, idx_n]);
                                helper_stack.push(self.topo_sort_find_neigh(idx_n));
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            stack.push(root);
                        }
                    }
                }
            }
        }
        stack
    }

    pub fn forward(&mut self, do_sort: bool) -> &mut Self {
        if do_sort {
            self.topo = self.topo_sort();
        }

        for t in &self.topo {
            let tensor = &self.tensors[*t];
            // Binary Operation
            if let (Some(l_idx), Some(r_idx), Some(op)) =
                (tensor.left_tref, tensor.right_tref, tensor.op)
            {
                if let (Some(l_value), Some(r_value)) =
                    (self.tensors[l_idx].value, self.tensors[r_idx].value)
                {
                    match op {
                        Operation::Add => self.tensors[*t].value = Some(l_value + r_value),
                        Operation::Sub => self.tensors[*t].value = Some(l_value - r_value),
                        Operation::Mult => self.tensors[*t].value = Some(l_value * r_value),
                        _ => break,
                    }
                }
                // Unary Operation
            } else if let (Some(l_idx), Some(op)) = (tensor.left_tref, tensor.op) {
                if let Some(l_value) = self.tensors[l_idx].value {
                    match op {
                        Operation::Pow2 => self.tensors[*t].value = Some(l_value.powi(2)),
                        Operation::Tanh => self.tensors[*t].value = Some(l_value.tanh()),
                        _ => break,
                    }
                }
            }
        }

        self
    }

    pub fn backward(&mut self, do_sort: bool) -> &mut Self {
        if do_sort {
            self.topo_reverse = self.topo_sort();
            self.topo_reverse.reverse();
        }

        for t in &self.topo_reverse {
            let tensor = &self.tensors[*t];

            // Binary Operation
            if let (Some(l_idx), Some(r_idx), Some(op), Some(grad)) =
                (tensor.left_tref, tensor.right_tref, tensor.op, tensor.grad)
            {
                match op {
                    Operation::Add => {
                        // y = x1 + x2
                        // d/dx1(y) = 1
                        // d/dx2(y) = 1
                        self.tensors[l_idx].grad =
                            Some(self.tensors[l_idx].grad.unwrap_or(0.0) + grad);
                        self.tensors[r_idx].grad =
                            Some(self.tensors[r_idx].grad.unwrap_or(0.0) + grad);
                    }
                    Operation::Sub => {
                        // y = x1 - x2
                        // d/dx1(y) = 1
                        // d/dx2(y) = -1
                        self.tensors[l_idx].grad =
                            Some(self.tensors[l_idx].grad.unwrap_or(0.0) + grad);
                        self.tensors[r_idx].grad =
                            Some(self.tensors[r_idx].grad.unwrap_or(0.0) - grad);
                    }
                    Operation::Mult => {
                        // y = x1 * x2
                        // d/dx1(y) = x2
                        // d/dx2(y) = x1
                        if let (Some(l_val), Some(r_val)) =
                            (self.tensors[l_idx].value, self.tensors[r_idx].value)
                        {
                            self.tensors[l_idx].grad =
                                Some(self.tensors[l_idx].grad.unwrap_or(0.0) + r_val * grad);
                            self.tensors[r_idx].grad =
                                Some(self.tensors[r_idx].grad.unwrap_or(0.0) + l_val * grad);
                        }
                    }
                    _ => break,
                }
            // Unary Operation
            } else if let (Some(l_idx), Some(op), Some(value), Some(grad)) =
                (tensor.left_tref, tensor.op, tensor.value, tensor.grad)
            {
                match op {
                    // y = x**2
                    // d/dx(y) = 2x
                    Operation::Pow2 => {
                        if let Some(l_val) = self.tensors[l_idx].value {
                            self.tensors[l_idx].grad = Some(2.0 * l_val * grad);
                        }
                    }
                    // y = tanh(x)
                    // d/dx(y) = 1 - y**2
                    Operation::Tanh => {
                        self.tensors[l_idx].grad = Some((1.0 - value.powi(2)) * grad);
                    }
                    _ => break,
                }
            }
        }
        self
    }

    pub fn gradient(&mut self, learning_rate: f64) -> &mut Self {
        for n in 0..self.neurons.len() {
            for w in 0..self.neurons[n].weights_tref.len() {
                let w = &mut self.tensors[self.neurons[n].weights_tref[w]];
                w.value = Some(w.value.unwrap_or(0.0) - learning_rate * w.grad.unwrap_or(0.0));
            }
            let bias = &mut self.tensors[self.neurons[n].bias_tref];
            bias.value = Some(bias.value.unwrap_or(0.0) - learning_rate * bias.grad.unwrap_or(0.0));
        }

        self
    }

    pub fn reset_grad(&mut self, idx: TensorRef) -> &mut Self {
        for i in 0..self.tensors.len() {
            self.tensors[i].grad = Some(0.0);
        }
        self.tensors[idx].grad = Some(1.0);
        self
    }

    pub fn nb_params(&mut self) -> usize {
        let mut count = 0;
        for n in 0..self.neurons.len() {
            for _ in 0..self.neurons[n].weights_tref.len() {
                count += 1;
            }
            count += 1;
        }
        count
    }

    pub fn summary(&mut self) -> &mut Self {
        let nbp = self.nb_params();

        println!("~~ Summary ~~");

        for l in &self.layers {
            let nin = l.nin;
            let nout = l.nout;
            println!("Layer({nin}x{nout})");

            for nref in &l.neurons_nref {
                let nin = self.neurons[*nref].nin;
                println!("\t TanhNeuron({nin})");
            }
        }
        println!("Number of parameters {nbp}");

        self
    }

    // Make clippy happy...
    #[allow(clippy::too_many_lines)]
    pub fn draw_helper(&mut self) -> String {
        let mut layers_def = String::new();
        let mut layer_name: &str;
        let mut layer_color: &str;

        for (idx, layer) in self.layers.iter().enumerate() {
            let mut tensors_def = String::new();

            for nref in &layer.neurons_nref {
                for wref in &self.neurons[*nref].weights_tref {
                    let tensor = &self.tensors[*wref];
                    tensors_def.push_str(&format!(
                        "tensor{}[label=\"{{ {} | {} | ∂ {} }}\", shape=record];\n",
                        tensor.curr_tref.unwrap_or(0),
                        tensor.label,
                        tensor.value.unwrap_or(0.0),
                        tensor.grad.unwrap_or(0.0),
                    ));
                }

                let tensor = &self.tensors[self.neurons[*nref].bias_tref];
                tensors_def.push_str(&format!(
                    "tensor{}[label=\"{{ {} | {} | ∂ {} }}\", shape=record];\n",
                    tensor.curr_tref.unwrap_or(0),
                    tensor.label,
                    tensor.value.unwrap_or(0.0),
                    tensor.grad.unwrap_or(0.0),
                ));

                for cref in &self.neurons[*nref].values_tref {
                    let tensor = &self.tensors[*cref];
                    tensors_def.push_str(&format!(
                        "tensor{}[label=\"{{ {} | {} | ∂ {} }}\", shape=record];\n",
                        tensor.curr_tref.unwrap_or(0),
                        tensor.label,
                        tensor.value.unwrap_or(0.0),
                        tensor.grad.unwrap_or(0.0),
                    ));
                }
            }

            if idx == 0 {
                layer_name = "Input Layer";
                layer_color = "mediumvioletred";
            } else if idx == self.layers.len() - 1 {
                layer_name = "Output Layer";
                layer_color = "mediumseagreen";
            } else {
                layer_name = "Hidden Layer";
                layer_color = "mediumslateblue";
            }

            layers_def.push_str(&format!(
                "subgraph cluster_L{} {{\n\
                color=white;\n\
                label = \"{} ({}x{})\";\n\
                node [style=solid,color={}, shape=circle];\n\
                {}}};\n",
                idx, layer_name, layer.nin, layer.nout, layer_color, tensors_def
            ));
        }

        let mut xis_def = String::new();
        for xv_ref in &self.xis_ref {
            for x_ref in xv_ref {
                let tensor = &self.tensors[*x_ref];
                xis_def.push_str(&format!(
                    "tensor{}[label=\"{{ {} | {} | ∂ {} }}\", shape=record];\n",
                    tensor.curr_tref.unwrap_or(0),
                    tensor.label,
                    tensor.value.unwrap_or(0.0),
                    tensor.grad.unwrap_or(0.0),
                ));
            }
        }
        xis_def = format!(
            "subgraph cluster_X {{\n\
            color=white;\n\
            label = \"X Data Layer\";\n\
            node [style=solid,color=mediumpurple4, shape=circle];\n\
            {xis_def}}};\n"
        );

        let mut xis_pred_def = String::new();
        for xv_pred_ref in &self.xis_pred_ref {
            for x_pred_ref in xv_pred_ref {
                let tensor = &self.tensors[*x_pred_ref];
                xis_pred_def.push_str(&format!(
                    "tensor{}[label=\"{{ {} | {} }}\", shape=record];\n",
                    tensor.curr_tref.unwrap_or(0),
                    tensor.label,
                    tensor.value.unwrap_or(0.0),
                ));
            }
        }
        xis_pred_def = format!(
            "subgraph cluster_XP {{\n\
            color=white;\n\
            label = \"X Prediction Layer\";\n\
            node [style=solid,color=mediumturquoise, shape=circle];\n\
            {xis_pred_def}}};\n"
        );

        let mut losses_def = String::new();
        for loss_ref in &self.losses_ref {
            let tensor = &self.tensors[*loss_ref];
            losses_def.push_str(&format!(
                "tensor{}[label=\"{{ {} | {} | ∂ {} }}\", shape=record];\n",
                tensor.curr_tref.unwrap_or(0),
                tensor.label,
                tensor.value.unwrap_or(0.0),
                tensor.grad.unwrap_or(0.0),
            ));
        }
        losses_def = format!(
            "subgraph cluster_Loss {{\n\
            color=white;\n\
            label = \"Loss Layer\";\n\
            node [style=solid,color=mediumorchid, shape=circle];\n\
            {losses_def}}};\n"
        );

        let mut tensorsdep = String::new();
        for t in self.topo_sort() {
            let tensor = &self.tensors[t];
            if let Some(tl) = tensor.left_tref {
                tensorsdep.push_str(&format!(
                    "tensor{}->tensor{};\n",
                    tl,
                    tensor.curr_tref.unwrap_or(0)
                ));
            }
            if let Some(rl) = tensor.right_tref {
                tensorsdep.push_str(&format!(
                    "tensor{}->tensor{};\n",
                    rl,
                    tensor.curr_tref.unwrap_or(0)
                ));
            }
        }

        format!(
            "digraph {{\n\
            graph [rankdir=LR];\n\
            {layers_def}{xis_def}{xis_pred_def}{losses_def}{tensorsdep}}}"
        )
    }

    // Make clippy happy...
    #[allow(clippy::missing_panics_doc)]
    pub fn draw(&mut self, filename: &str) -> &mut Self {
        let graph = self.draw_helper();
        let _ = write!(File::create(filename).unwrap(), "{graph}");

        self
    }
}

pub struct NeuralNet {
    en: Engine,
}

impl Default for NeuralNet {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralNet {
    #[must_use]
    pub fn new() -> Self {
        NeuralNet { en: Engine::new() }
    }

    pub fn model(&mut self, layers: &[usize]) -> &mut Self {
        self.en.multilayer(layers);
        self
    }

    pub fn make_x_tensor(&mut self, xis: &[Vec<f64>]) -> Vec<Vec<TensorRef>> {
        let mut xis_tensor = vec![];

        for (i, xv) in xis.iter().enumerate() {
            let mut xv_tensor = vec![];
            for (j, v) in xv.iter().enumerate() {
                xv_tensor.push(self.en.tensor(*v, format!("X{i}{j}").as_str()));
            }
            xis_tensor.push(xv_tensor);
        }
        xis_tensor
    }

    pub fn make_y_tensor(&mut self, y_target: &[f64]) -> Vec<TensorRef> {
        let mut y_target_tensor = vec![];

        for (i, v) in y_target.iter().enumerate() {
            let tensor_ref = self.en.tensor(*v, format!("ytarget{i}").as_str());
            y_target_tensor.push(tensor_ref);
            self.en.losses_ref.push(tensor_ref);
        }
        y_target_tensor
    }

    pub fn train(
        &mut self,
        xis: &[Vec<f64>],
        y_target: &[f64],
        nb_iter: usize,
        learning_rate: f64,
        debug: bool,
    ) -> &mut Self {
        let xis_tensor = self.make_x_tensor(xis);
        self.en.xis_ref = xis_tensor.clone();

        let y_target_tensor = self.make_y_tensor(y_target);
        let mut y_pred_tensor = vec![];

        for x in xis_tensor {
            y_pred_tensor.push(self.en.multilayer_connect(&x));
        }

        let loss = self.en.tensor_loss(&y_target_tensor, &y_pred_tensor);

        let mut do_sort = true;

        for i in 0..nb_iter {
            self.en
                .forward(do_sort)
                .reset_grad(loss)
                .backward(do_sort)
                .gradient(learning_rate);
            do_sort = false;

            if debug {
                println!("Step {i}: Loss {:?}", self.en.tensor_value(loss));
            }
        }

        self
    }

    pub fn predict(&mut self, xis: &[Vec<f64>]) -> Vec<f64> {
        // Improvement 1: create a new model for prediction using the training weights (avoid forward recompute)
        // Improvement 2: don't create a new X tensor each time, reuse the previous one (reduce the memory size)

        let xis_tensor = self.make_x_tensor(xis);
        self.en.xis_pred_ref = xis_tensor.clone();

        let mut y_pred_tensor = vec![];

        for x in xis_tensor {
            y_pred_tensor.push(self.en.multilayer_connect(&x));
        }

        self.en.forward(true);

        let mut res = vec![];
        for r in y_pred_tensor {
            res.push(self.en.tensor_value(r));
        }

        res
    }

    pub fn summary(&mut self) -> &mut Self {
        self.en.summary();
        self
    }

    pub fn draw(&mut self, filename: &str) -> &mut Self {
        self.en.draw(filename);
        self
    }
}

use pyo3::prelude::*;

#[pyfunction]
fn hello() {
    println!("quack!");
}

#[pyclass(name = "NeuralNet")]
pub struct NeuralNetPy {
    nn: NeuralNet,
}

#[allow(clippy::new_without_default)]
#[pymethods]
impl NeuralNetPy {
    #[must_use]
    #[new]
    pub fn new() -> Self {
        NeuralNetPy {
            nn: NeuralNet::new(),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn model(mut slf: PyRefMut<'_, Self>, layers: Vec<usize>) -> PyRefMut<'_, Self> {
        slf.nn.en.multilayer(&layers);
        slf
    }

    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn train(
        mut slf: PyRefMut<'_, Self>,
        xis: Vec<Vec<f64>>,
        y_target: Vec<f64>,
        nb_iter: usize,
        learning_rate: f64,
        debug: bool,
    ) -> PyRefMut<'_, Self> {
        slf.nn.train(&xis, &y_target, nb_iter, learning_rate, debug);
        slf
    }

    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn predict(mut slf: PyRefMut<'_, Self>, xis: Vec<Vec<f64>>) -> Vec<f64> {
        slf.nn.predict(&xis)
    }

    #[must_use]
    pub fn summary(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.nn.summary();
        slf
    }

    #[must_use]
    pub fn draw<'a>(mut slf: PyRefMut<'a, Self>, filename: &'a str) -> PyRefMut<'a, Self> {
        slf.nn.draw(filename);
        slf
    }
}

#[pymodule]
fn kanars(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_class::<NeuralNetPy>()?;
    Ok(())
}

#[test]
fn test_forward_basic() {
    let mut en = Engine::new();
    let x1 = en.tensor(3.0, "");
    let y1 = en.tensor(5.0, "");
    let z1 = en.tensor_add(x1, y1, "");
    let z2 = en.tensor_mult(z1, y1, "");

    en.forward(true);

    assert_eq!(en.tensors[z2].value.unwrap(), 40.0);
}

#[test]
fn test_forward_ops() {
    let mut en = Engine::new();
    let x1 = en.tensor(2.0, "x1");
    let x2 = en.tensor(0.0, "x2");
    let w1 = en.tensor(-3.0, "w1");
    let w2 = en.tensor(1.0, "w2");
    let b1 = en.tensor(6.881_373, "b1");
    let x1w1 = en.tensor_mult(x1, w1, "x1*w1");
    let x2w2 = en.tensor_mult(x2, w2, "x2*w2");
    let x1w1x2w2 = en.tensor_add(x1w1, x2w2, "x1*w1+x2*w2");
    let n1 = en.tensor_add(x1w1x2w2, b1, "n1");
    let o1 = en.tensor_tanh(n1, "o1");

    en.forward(true);

    assert!((0.70710..0.70711).contains(&en.tensors[o1].value.unwrap()))
}

#[test]
fn test_backward_topo_sort() {
    let mut en = Engine::new();
    let v0 = en.tensor(0.0, "v0");
    let v1 = en.tensor(1.0, "v1");
    let v3 = en.tensor_tanh(v1, "v3");
    let v2 = en.tensor_tanh(v3, "v2");
    let v4 = en.tensor_add(v0, v1, "v4");
    let v5 = en.tensor_add(v2, v0, "v5");
    let _v6 = en.tensor_add(v5, v4, "v6");
    let expected = ["v0", "v1", "v3", "v2", "v4", "v5", "v6"];

    let topo = en.forward(true).topo_sort();

    for i in topo {
        assert_eq!(en.tensors[i].label, expected[i]);
    }
}

#[test]
fn test_backward_basic() {
    let mut en = Engine::new();
    let x1 = en.tensor(2.0, "x1");
    let x2 = en.tensor(0.0, "x2");
    let w1 = en.tensor(-3.0, "w1");
    let w2 = en.tensor(1.0, "w2");
    let b1 = en.tensor(6.881_373, "b1");
    let x1w1 = en.tensor_mult(x1, w1, "x1*w1");
    let x2w2 = en.tensor_mult(x2, w2, "x2*w2");
    let x1w1x2w2 = en.tensor_add(x1w1, x2w2, "x1*w1+x2*w2");
    let n1 = en.tensor_add(x1w1x2w2, b1, "n1");
    let o1 = en.tensor_tanh(n1, "o1");

    en.forward(true).reset_grad(o1).backward(true);

    assert_eq!(en.tensors[w2].grad.unwrap(), 0.0);
    assert!((-1.5001..-1.5000).contains(&en.tensors[x1].grad.unwrap()));
    assert!((0.5000..0.5001).contains(&en.tensors[x2].grad.unwrap()));
    assert!((1.0000..1.0001).contains(&en.tensors[w1].grad.unwrap()));
}

#[test]
fn test_backward_multref() {
    let mut en = Engine::new();
    let a1 = en.tensor(-2.0, "a1");
    let b1 = en.tensor(3.0, "b1");
    let d1 = en.tensor_mult(a1, b1, "d1");
    let e1 = en.tensor_add(a1, b1, "e1");
    let f1 = en.tensor_mult(d1, e1, "f1");

    en.forward(true).reset_grad(f1).backward(true);

    assert_eq!(en.tensors[a1].grad.unwrap(), -3.0);
    assert_eq!(en.tensors[b1].grad.unwrap(), -8.0);
}

#[test]
fn test_neuron() {
    let mut en = Engine::new();
    let xis = [en.tensor(2.0, "x1"), en.tensor(3.0, "x2")];
    let n1 = en.neuron(2);
    let res = en.neuron_connect(n1, &xis);

    en.forward(true);

    assert!((0.9091..0.9092).contains(&en.tensor_value(res)));
}

#[test]
fn test_layer() {
    let mut en = Engine::new();
    let xis = [en.tensor(2.0, "x1"), en.tensor(3.0, "x2")];
    let l1 = en.layer(2, 3);
    let res = en.layer_connect(l1, &xis);

    en.forward(true);

    assert_eq!(res, [15, 20, 25]);
    for r in res {
        assert!((0.9091..0.9092).contains(&en.tensor_value(r)));
    }
}

#[test]
fn test_multilayer() {
    let mut en = Engine::new();
    let xis = [
        en.tensor(2.0, "x1"),
        en.tensor(3.0, "x2"),
        en.tensor(-1.0, "x3"),
    ];

    en.multilayer(&[3, 4, 4, 1]);
    let res = en.multilayer_connect(&xis);
    en.forward(true);

    assert!((0.0011..0.0012).contains(&en.tensor_value(res)));
}

#[test]
fn test_multilayer_forward_loss() {
    let mut en = Engine::new();
    let xs = [
        [
            en.tensor(2.0, "x1"),
            en.tensor(3.0, "x2"),
            en.tensor(-1.0, "x3"),
        ],
        [
            en.tensor(3.0, "x4"),
            en.tensor(-1.0, "x5"),
            en.tensor(0.5, "x6"),
        ],
        [
            en.tensor(0.5, "x7"),
            en.tensor(1.0, "x8"),
            en.tensor(1.0, "x9"),
        ],
        [
            en.tensor(1.0, "x10"),
            en.tensor(1.0, "x11"),
            en.tensor(-1.0, "x12"),
        ],
    ];
    let ytarget = [
        en.tensor(1.0, "y1"),
        en.tensor(-1.0, "y2"),
        en.tensor(-1.0, "y3"),
        en.tensor(1.0, "y4"),
    ];

    en.multilayer(&[3, 4, 4, 1]);
    let ypred = [
        en.multilayer_connect(&xs[0]),
        en.multilayer_connect(&xs[1]),
        en.multilayer_connect(&xs[2]),
        en.multilayer_connect(&xs[3]),
    ];
    let loss = en.tensor_loss(&ytarget, &ypred);
    en.forward(true);

    assert!((0.0011..0.0012).contains(&en.tensor_value(ypred[0])));
    assert!((-0.0013..-0.0012).contains(&en.tensor_value(ypred[1])));
    assert!((0.0007..0.0008).contains(&en.tensor_value(ypred[2])));
    assert!((0.0003..0.0004).contains(&en.tensor_value(ypred[3])));
    assert!((3.9961..3.9962).contains(&en.tensor_value(loss)));
}

#[test]
fn test_multilayer_backward_loss() {
    let mut en = Engine::new();
    let xis = [
        [
            en.tensor(2.0, "x1"),
            en.tensor(3.0, "x2"),
            en.tensor(-1.0, "x3"),
        ],
        [
            en.tensor(3.0, "x4"),
            en.tensor(-1.0, "x5"),
            en.tensor(0.5, "x6"),
        ],
        [
            en.tensor(0.5, "x7"),
            en.tensor(1.0, "x8"),
            en.tensor(1.0, "x9"),
        ],
        [
            en.tensor(1.0, "x10"),
            en.tensor(1.0, "x11"),
            en.tensor(-1.0, "x12"),
        ],
    ];

    en.multilayer(&[3, 4, 4, 1]);
    let res1 = en.multilayer_connect(&xis[0]);
    let res2 = en.multilayer_connect(&xis[1]);
    let res3 = en.multilayer_connect(&xis[2]);
    let res4 = en.multilayer_connect(&xis[3]);
    let ytarget = [
        en.tensor(1.0, "y1"),
        en.tensor(-1.0, "y2"),
        en.tensor(-1.0, "y3"),
        en.tensor(1.0, "y4"),
    ];
    let ypred = [res1, res2, res3, res4];
    let loss = en.tensor_loss(&ytarget, &ypred);
    en.forward(true);
    en.reset_grad(loss);
    en.backward(true);

    for i in 0..100 {
        en.forward(false)
            .reset_grad(loss)
            .backward(false)
            .gradient(0.05);
        if i == 0 {
            assert!((3.9961..3.9962).contains(&en.tensor_value(loss)));
        } else if i == 99 {
            assert!((0.0091..0.0092).contains(&en.tensor_value(loss)));
        }
    }
}

#[test]
fn test_multilayer_more_predict() {
    let mut en = Engine::new();
    let xis = [
        [en.tensor(2.0, "x1"), en.tensor(6.0, "x2")],
        [en.tensor(4.0, "x3"), en.tensor(2.0, "x4")],
        [en.tensor(6.0, "x5"), en.tensor(10.0, "x6")],
        [en.tensor(8.0, "x7"), en.tensor(10.0, "x8")],
        [en.tensor(10.0, "x9"), en.tensor(4.0, "x10")],
    ];
    en.multilayer(&[2, 4, 4, 1]);
    let res1 = en.multilayer_connect(&xis[0]);
    let res2 = en.multilayer_connect(&xis[1]);
    let res3 = en.multilayer_connect(&xis[2]);
    let res4 = en.multilayer_connect(&xis[3]);
    let res5 = en.multilayer_connect(&xis[4]);
    let ypred = [res1, res2, res3, res4, res5];
    let ytarget = [
        en.tensor(1.0, "y1"),
        en.tensor(-1.0, "y2"),
        en.tensor(1.0, "y3"),
        en.tensor(1.0, "y4"),
        en.tensor(-1.0, "y5"),
    ];
    let loss = en.tensor_loss(&ytarget, &ypred);

    for _ in 0..10 {
        en.forward(true)
            .reset_grad(loss)
            .backward(true)
            .gradient(0.05);
    }

    assert!((0.0576..0.0577).contains(&en.tensor_value(loss)));

    let xis_new = [
        [en.tensor(5.0, "x1"), en.tensor(9.0, "x2")],
        [en.tensor(7.0, "x3"), en.tensor(3.0, "x4")],
    ];

    let ypred_new1 = en.multilayer_connect(&xis_new[0]);
    let ypred_new2 = en.multilayer_connect(&xis_new[1]);

    en.forward(true);

    assert!((0.9048..0.9049).contains(&en.tensor_value(ypred_new1)));
    assert!((-0.9076..-0.9075).contains(&en.tensor_value(ypred_new2)));

    assert_eq!(en.nb_params(), 37);
}

#[test]
fn test_neuralnet_linear() {
    let mut nn = NeuralNet::new();
    let xis = [
        vec![2.0, 6.0],
        vec![4.0, 2.0],
        vec![6.0, 10.0],
        vec![8.0, 10.0],
        vec![10.0, 4.0],
    ];

    let y_target = [1.0, -1.0, 1.0, 1.0, -1.0];
    let res = nn
        .model(&[2, 4, 4, 1])
        .train(&xis, &y_target, 10, 0.05, true)
        .predict(&[vec![5.0, 9.0], vec![7.0, 3.0]]);

    assert!((0.90..1.0).contains(&res[0]));
    assert!((-1.0..-0.90).contains(&res[1]));
}

#[test]
fn test_neuralnet_draw_summary() {
    let mut nn = NeuralNet::new();
    let xis = [
        vec![2.0, 6.0],
        vec![4.0, 2.0],
        vec![6.0, 10.0],
        vec![8.0, 10.0],
        vec![10.0, 4.0],
    ];

    let y_target = [1.0, -1.0, 1.0, 1.0, -1.0];
    nn.model(&[2, 4, 4, 1])
        .train(&xis, &y_target, 10, 0.05, true)
        .predict(&[vec![5.0, 9.0], vec![7.0, 3.0]]);

    // Check no crashs
    nn.summary();
    nn.draw("test_kanars.dot");
}

#[test]
fn test_neuralnet_circle() {
    // Circle equation: x**2 + y**2 = 5**2

    let mut nn = NeuralNet::new();
    let xis = &[
        // Inside
        vec![0.0, 4.0],
        vec![2.0, 2.0],
        vec![2.0, -1.0],
        vec![-2.0, 0.0],
        vec![0.0, 0.0],
        vec![1.0, 4.0],
        vec![2.5, 0.0],
        vec![-4.0, 2.0],
        vec![-3.5, -1.0],
        vec![-2.5, -2.5],
        // Outside
        vec![0.0, 8.0],
        vec![6.0, 1.0],
        vec![-8.0, 6.0],
        vec![-4.0, 7.5],
        vec![-6.0, -7.0],
        vec![5.0, 5.0],
        vec![0.0, -6.0],
        vec![4.0, -5.0],
        vec![-8.0, -2.5],
        vec![4.5, 7.5],
    ];

    let y_target = [
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
    ];
    let res = nn
        .model(&[2, 4, 4, 1])
        .train(xis, &y_target, 1000, 0.01, true)
        .predict(&[vec![1.5, 3.0], vec![-7.0, -8.0]]);

    assert!((-1.0..-0.95).contains(&res[0]));
    assert!((0.95..1.0).contains(&res[1]));
}
