mod F2;

#[macro_use]
extern crate log;
extern crate core;

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};
use log::{info, warn};
use std::fmt::Write;
use std::path::Path;
use std::string::String;
use rand::Rng;

// Json representation of the data struct
#[derive(Serialize, Deserialize, Debug, Default)]
struct GrammarJson(HashMap<String, Vec<Vec<String>>>);

// Fragment id unique for each fragment
#[derive(Clone, Debug, Copy, Default, PartialEq)]
struct FragmentId(usize);

// Fragment data structure
#[derive(Clone, Debug)]
enum Fragment {
    // nonterminal contains a vector of fragments (some might be non-terminal), probabilities, probabilities sum and steps
    NonTerminal(Vec<FragmentId>, Vec<FragmentId>, u32, Vec<FragmentId>),
    // Ordered list of fragments, probabilities and number of steps to a terminal
    Expression(Vec<FragmentId>, u32),
    // terminal results to bytes
    Terminal(Vec<u8>),
    // empty expressions - no operation
    Nop,
}

// Rust representation: transformed into nested structure
#[derive(Debug, Default)]
struct Grammar {
    // all fragments in grammar
    fragments: Vec<Fragment>,

    // starting fragment
    start: String,

    // mapping of non-terminal names to fragment identifiers
    name_to_fragment: BTreeMap<String, FragmentId>,

    // Xorshift seed
    // in cell so that we do not need mutable access
    // https://doc.rust-lang.org/std/cell/
    seed: Cell<usize>,
}

impl Grammar {
    fn new(grammar_json: &GrammarJson) -> Self {
        let mut grammar = Grammar::default();

        grammar.start = String::from("<start>");

        // allocate all non-terminals
        for (f_name, _) in &grammar_json.0 {
            if grammar.name_to_fragment.contains_key(f_name) {
                panic!("{} fragment declaration repeats in grammar.", f_name);
            }
            // allocate a new empty fragment
            let fragment_id = grammar.allocate_fragment(
                Fragment::NonTerminal(
                    Vec::new(), Vec::new(), 0, Vec::new()));
            debug!("Allocated non-term {} with id {:?}", f_name, fragment_id);

            grammar.name_to_fragment.insert(
                f_name.clone(),
                fragment_id,
            );
        }

        // check if grammar contains the start token
        if !grammar.name_to_fragment.contains_key(&*grammar.start) {
            panic!("Grammar does not contain the \"<start>\" fragment.")
        }

        // traverse the grammar to create the data structure
        for (f_name, f_options) in &grammar_json.0 {
            // all options for a fragment and their probabilities
            let mut options = Vec::new();
            let mut probabilities = Vec::new();
            let mut probabilities_sum : usize = 0;

            // for each option - wrapped as expression due to json formatting
            for expression in f_options {
                let mut probability: Option<usize> = None;

                // check if it has a probability
                match expression.last() {
                    Some(str) => {
                        if str.starts_with("p=") {
                            let num = String::from(&str[2..]).parse::<u32>();
                            match num {
                                Ok(ok) => { probability = Some(ok as usize); }
                                Err(_) => {
                                    panic!(
                                        "A fragment {} contains an option with badly formatted probability field!"
                                        , f_name
                                    );
                                }
                            }
                        }
                    }
                    None => {}
                }

                let mut subfragments: Vec<FragmentId> = Vec::new();

                // don't iterate through probability as a term in expression
                let iter = if probability.is_some() { expression.len() - 1 } else { expression.len() };
                // default probability - 1
                probability = if probability.is_some() { probability } else { Some(1) };
                probabilities_sum += if probability.is_some() {probability.unwrap()} else { 0 };
                probabilities.push(probability.unwrap());

                // for each term in an expression
                for index in 0..iter {
                    let subfragment = expression.get(index).unwrap();
                    let mut subfragment_id = FragmentId(0);

                    // if non-term
                    if subfragment.starts_with("<") {
                        if subfragment.ends_with(">") {
                            // if "<>", it must be a non-terminal with a corresponding body
                            if !grammar.name_to_fragment.contains_key(subfragment) {
                                panic!("subfragment {} in fragment {} is never defined.",
                                       subfragment, f_name);
                            } else {
                                // stored non-terminal
                                subfragment_id = *grammar.name_to_fragment.get(subfragment).unwrap();
                            }
                        }
                    }

                    // else - term
                    else {
                        if subfragment.is_empty() {
                            // store empty terminal
                            subfragment_id = grammar.allocate_fragment(
                                Fragment::Nop);
                            debug!("\t Added NOP {} to grammar", subfragment);
                        }

                        // store non-terminal
                        if !grammar.name_to_fragment.contains_key(subfragment) {
                            // store terminal
                            subfragment_id = grammar.allocate_fragment(
                                Fragment::Terminal(subfragment.as_bytes().to_vec()));
                            debug!("\t Added terminal {} to grammar", subfragment);
                        }
                    }
                    subfragments.push(subfragment_id);
                }

                options.push(
                    grammar.allocate_fragment(
                        Fragment::Expression(subfragments, u32::MAX)
                    ));
            }

            let mut probability_fragments = Vec::new();
            if probabilities_sum == probabilities.len(){
                probability_fragments = options.clone();
            }else {
                println!("Probabilities in fragment {}!!!", f_name);
                for (index, p) in probabilities.iter().enumerate(){
                    for _ in 0..*p {
                        probability_fragments.push(options[index]);
                    }
                }
            }
            // reallocate the updated version of the non-terminal with body onto the vector
            grammar.reallocate_nonterm(
                Fragment::NonTerminal(options, probability_fragments, u32::MAX,  Vec::new()),
                f_name,
            );
        }

        grammar.optimise();
        grammar.find_steps();
        grammar.optimise_stepcount();

        grammar
    }

    fn find_steps(&mut self) -> u32 {
        // think of the grammar as a connected graph, traverse each layer
        // depth first search to assign steps to terminal for each fragment

        println!();
        println!("*******************************************************************");
        println!("*                               STEPS                             *");
        println!("*******************************************************************");
        println!();

        // initially, start node
        let mut stack: VecDeque<FragmentId> = VecDeque::new();
        stack.push_back(*self.name_to_fragment.get(&self.start).unwrap());

        // nodes that pushed children onto stack
        let mut visited: HashSet<usize> = HashSet::new();
        // pushed back because of a cycle
        let mut pushed: HashSet<usize> = HashSet::new();

        // traverse entire graph
        while !stack.is_empty() {
            // debug!("STACK CONTAINS: {:?}", stack);
            let curr_id = stack.back().unwrap().0;
            let curr = self.fragments.get(curr_id).unwrap();

            if visited.contains(&curr_id) {
                // process visited nodes
                stack.pop_back();
                match curr {
                    // for non-term - min+1 = min_steps
                    Fragment::NonTerminal(options, probabilities, ..) => {
                        let mut min = u32::MAX;
                        let mut shortest = Vec::new();
                        for o in options {
                            let option = self.fragments.get(o.0).unwrap();
                            match option {
                                Fragment::Terminal(_) => {
                                    if min > 1 {
                                        min = 1;
                                        shortest.clear();
                                    }

                                    shortest.push(*o);
                                }
                                Fragment::NonTerminal(_, _,  s, ..) => {
                                    if (*s) < min {
                                        min = *s + 1;
                                        shortest.clear();
                                        shortest.push(*o);
                                    } else if (*s) == min {
                                        shortest.push(*o);
                                    }
                                }
                                Fragment::Expression(_, s) => {
                                    if (*s) < min {
                                        min = *s + 1;
                                        shortest.clear();
                                        shortest.push(*o);
                                    } else if (*s) == min {
                                        shortest.push(*o);
                                    }
                                }
                                Fragment::Nop => {
                                    if min > 0 {
                                        min = 0;
                                        shortest.clear();
                                    }

                                    shortest.push(*o);
                                }
                            }
                        }
                        // if all children have cycles, push back
                        if min == u32::MAX {
                            // if it has already pushed back - inconclusive cycle
                            // can't calculate min steps.
                            if !pushed.contains(&curr_id) {
                                stack.push_front(FragmentId(curr_id));
                                pushed.insert(curr_id);
                            } else {
                                debug!("No way to calculate the shortest for non-terminal: {}", curr_id);
                                // no way to calculate shortest so all options are
                                self.fragments[curr_id] =
                                    Fragment::NonTerminal(options.clone(),
                                                          probabilities.clone(),
                                                          min,
                                                          options.clone())
                            }
                        } else {
                            debug!("Calculated shortest steps for non-terminal {}, steps: {}, no.of shortest: {}", curr_id, min, shortest.len());
                            // self.print_fragment(FragmentId(curr_id));
                            self.fragments[curr_id] =
                                Fragment::NonTerminal(options.clone(),
                                                      probabilities.clone(),
                                                      min,
                                                      shortest.clone())
                        }
                    }
                    // for expression shortest path is sum of all children's shortest
                    Fragment::Expression(terms, mut _steps) => {
                        let mut min = 0;
                        // if a child is dependent on a parent this will be true: cannot calculate minimum
                        let mut uncalculated_child = false;
                        for t in terms {
                            let term = self.fragments.get(t.0).unwrap();
                            match term {
                                Fragment::Terminal(_) => min += 1,
                                Fragment::NonTerminal(_, _, s, _) => {
                                    if *s == u32::MAX {
                                        uncalculated_child = true;
                                    } else {
                                        min += *s + 1;
                                    }
                                }
                                Fragment::Expression(_, s) => {
                                    if *s == u32::MAX {
                                        uncalculated_child = true;
                                    } else {
                                        min += *s + 1;
                                    }
                                }
                                // skip NOPs
                                _ => {}
                            }
                        }
                        min = if uncalculated_child { u32::MAX } else { min };
                        if uncalculated_child {
                            min = u32::MAX;
                            if !pushed.contains(&curr_id) {
                                stack.push_front(FragmentId(curr_id));
                                pushed.insert(curr_id);
                            } else {
                                // can't calculate min steps
                                debug!("Can't calculate min_steps for expression {}", curr_id);
                                _steps = min;
                            }
                        } else {
                            _steps = min;
                            debug!("Calculated min steps for expression {}: {}!", curr_id, min);
                        }
                    }
                    _ => {
                        warn!("encountered a Term while assigning steps.")
                    }
                }
            } else {
                // if non-term or expression, push children
                match curr {
                    Fragment::NonTerminal(fragments, ..) => {
                        visited.insert(curr_id);
                        for f in fragments {
                            match self.fragments[f.0] {
                                Fragment::Terminal(_, ..) => {}
                                Fragment::Nop => {}
                                _ => {
                                    if !visited.contains(&f.0) && !stack.contains(&f) {
                                        stack.push_back(*f);
                                    }
                                }
                            }
                        }
                    }
                    Fragment::Expression(fragments, _) => {
                        visited.insert(curr_id);
                        for f in fragments {
                            match self.fragments[f.0] {
                                Fragment::Terminal(_, ..) => {}
                                Fragment::Nop => {}
                                _ => {
                                    if !visited.contains(&f.0) && !stack.contains(&f) {
                                        stack.push_back(*f);
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }


        0
    }

    fn optimise(&mut self) {
        println!();
        println!("*******************************************************************");
        println!("*                        OPTIMISE GRAMMAR                         *");
        println!("*******************************************************************");
        println!();
        // Keeps track of fragment identifiers which resolve to nops
        let mut nop_fragments = BTreeSet::new();

        // Track if a optimization had an effect
        let mut changed = true;
        while changed {
            // Start off assuming no effect from optimzation
            changed = false;

            // Go through each fragment, looking for potential optimizations
            for idx in 0..self.fragments.len() {
                // Clone the fragment such that we can inspect it, but we also
                // can mutate it in place.
                match self.fragments[idx].clone() {
                    Fragment::NonTerminal(options, ..) => {
                        // If this non-terminal only has one option, replace
                        // itself with the only option it resolves to
                        if options.len() == 1 {
                            self.fragments[idx] =
                                self.fragments[options[0].0].clone();
                            changed = true;
                        }
                    }
                    Fragment::Expression(expr, ..) => {
                        // If this expression doesn't have anything to do at
                        // all. Then simply replace it with a `Nop`
                        if expr.len() == 0 {
                            self.fragments[idx] = Fragment::Nop;
                            changed = true;

                            // Track that this fragment identifier now resolves
                            // to a nop
                            nop_fragments.insert(idx);
                        }

                        // If this expression only does one thing, then replace
                        // the expression with the thing that it does.
                        if expr.len() == 1 {
                            self.fragments[idx] =
                                self.fragments[expr[0].0].clone();
                            changed = true;
                        }

                        // Remove all `Nop`s from this expression, as they
                        // wouldn't result in anything occuring.
                        if let Fragment::Expression(exprs, ..) =
                        &mut self.fragments[idx] {
                            // Only retain fragments which are not nops
                            exprs.retain(|x| {
                                if nop_fragments.contains(&x.0) {
                                    // Fragment was a nop, remove it
                                    changed = true;
                                    false
                                } else {
                                    // Fragment was fine, keep it
                                    true
                                }
                            });
                        }
                    }
                    Fragment::Terminal(_) | Fragment::Nop => {
                        // Already maximally optimized
                    }
                }
            }
        }
    }

    /**
        Optimisation goes as follows:
        - if a non-terminal A has a single shortest path to non-terminal B
        - if non-terminal B has a single shortest path to non-terminal C
        - then non-terminals A's and B's shortest paths should lead directly to non-terminal C
        - hence skipping one redundant step
        - same principle follows for a longer chain of single shortest path non-terminals
    **/
    fn optimise_stepcount(&mut self) {
        println!();
        println!("*******************************************************************");
        println!("*                        OPTIMISE STEPCOUNT                       *");
        println!("*******************************************************************");
        println!();

        // initially, start node
        // stack to store fragments
        let mut stack: VecDeque<FragmentId> = VecDeque::new();
        stack.push_back(*self.name_to_fragment.get(&self.start).unwrap());
        // store fragments that have been in the stack
        let mut visited: HashSet<usize> = HashSet::new();
        // store fragments that have been in a chain before
        let mut chained: HashSet<usize> = HashSet::new();
        let mut visited_nonterms = 0;

        while !stack.is_empty() {
            // discover fragments with single shortest path
            // pop fragment from top of the stack
            let parent_id = stack.pop_back().unwrap().0;
            let parent = self.fragments.get(parent_id).unwrap();
            match parent {
                Fragment::NonTerminal(options, .., shortest) => {
                    visited_nonterms += 1;
                    debug!("Number of visited nonterms: {}", visited_nonterms);

                    // if more than 1 options then add them to the stack to further explore
                    for option in options {
                        // get option fragment
                        let option_fragment = self.fragments.get(option.0).unwrap();
                        match option_fragment {
                            // push non-terminals to the stack
                            // push expression to the stack, unwrap when popped
                            Fragment::NonTerminal(..) | Fragment::Expression(..) => {
                                if !visited.contains(&option.0) {
                                    stack.push_back(*option);
                                    visited.insert(option.0);
                                    // debug!("Adding to the stack:");
                                    // self.print_fragment(option.clone());
                                }
                            }
                            _ => {
                                // don't push back NOPs and terminals - nothing to optimise
                            }
                        }
                    }
                    // if 1 option then potential for optimisation
                    // if it has been in a chain, it already is optimised
                    // DFS for a chain
                    if shortest.len() == 1 && !chained.contains(&parent_id) {
                        debug!("Found a single shortest option non-terminal {}", parent_id);
                        // search for non-terminals to skip
                        // add all of single path non-terminal chain to the vector
                        let mut chain: Vec<usize> = Vec::new();
                        let mut curr = parent_id;
                        let mut chain_broke = false;
                        while !chain_broke {
                            chain.push(curr.clone());
                            debug!("Adding non-terminal {} to the chain.", curr);
                            // get next in chain
                            if let Fragment::NonTerminal(.., prev_shortest) =
                            self.fragments.get(curr.clone()).unwrap().clone() {
                                curr = prev_shortest[0].0
                            }
                            debug!("Next curr: {}", curr);

                            let curr_fragment = self.fragments.get(curr.clone()).unwrap();
                            match curr_fragment {
                                // if next in chain is a non-terminal
                                Fragment::NonTerminal(.., curr_shortest) => {
                                    if curr_shortest.len() != 1 {
                                        chain_broke = true;
                                        debug!("fragment {curr} broke the chain!");
                                        self.print_fragment(FragmentId(curr));
                                    }
                                    if !visited.contains(&curr){
                                        // add the non-term to the stack for further exploration
                                        stack.push_back(FragmentId(curr));
                                    }
                                }
                                _ => {
                                    chain_broke = true;
                                }
                            }
                        }
                        debug!("Total chain size: {}", chain.len());
                        debug!("Chain: {:?}", chain);
                        for id in &chain {
                            self.print_fragment(FragmentId(*id));
                        }
                        if chain.len() > 1 {
                            debug!("Min id for chain: {}", curr);
                            self.print_fragment(FragmentId(curr));
                            let new_shortest = Vec::from([FragmentId(curr)]);
                            // change shortest path of each fragment in the chain
                            while !chain.is_empty() {
                                let curr_id = chain.pop().unwrap();
                                chained.insert(curr_id);
                                let f = self.fragments.get(curr_id).unwrap();
                                match f {
                                    Fragment::NonTerminal(opt, probs, steps, curr_shortest) => {
                                        debug!("curr_shortest for {} changed from {} to {}!", curr_id, curr_shortest[0].0, curr);
                                        self.fragments[curr_id] =
                                            Fragment::NonTerminal(opt.clone(),
                                                                  probs.clone(),
                                                                  steps.clone(),
                                                                  new_shortest.clone());
                                    }
                                    _ => panic!()
                                }
                            }
                        }
                    }
                }
                // Unwrap expression, only push another expression to unwrap or a terminal
                Fragment::Expression(terms, ..) => {
                    for term in terms {
                        let term_fragment = self.fragments.get(term.0).unwrap();
                        match term_fragment {
                            Fragment::NonTerminal(..) | Fragment::Expression(..) => {
                                if !visited.contains(&term.0) {
                                    stack.push_back(*term);
                                    visited.insert(term.0);
                                }
                            }
                            _ => {
                                // don't push NOPs and terminals
                            }
                        }
                    }
                }
                _ => {
                    panic!("Error in optimise stepcount: Fragment {:?} shouldn't be in the stack...", parent);
                }
            }
        }
    }

    fn allocate_fragment(&mut self, fragment: Fragment) -> FragmentId {
        let id = self.fragments.len();
        self.fragments.push(fragment);
        FragmentId(id)
    }

    // overwrite the fragment with new information
    fn reallocate_nonterm(&mut self, fragment: Fragment, f_name: &String) -> FragmentId {
        let mut id;
        if self.name_to_fragment.contains_key(f_name) {
            id = *self.name_to_fragment.get(f_name).unwrap();
            match self.fragments.get(id.0).unwrap() {
                Fragment::NonTerminal(..) => {
                    self.fragments[id.0] = fragment;
                }
                _ => { panic!("No such non_term!") }
            }
        } else {
            id = self.allocate_fragment(fragment);
        }

        id
    }

    fn print_fragment(&self, id: FragmentId) {
        let f = self.fragments.get(id.0).unwrap();
        let mut f_print = String::new();

        match f {
            Fragment::Expression(x, _) => {
                write!(&mut f_print, "Expression {} with {} items: \n", id.0, x.len());
                for e in x {
                    let f_e = self.fragments.get(e.0).unwrap();
                    match f_e {
                        Fragment::Terminal(term) => {
                            write!(&mut f_print, "\tterm \"{}\"\n", std::str::from_utf8(term).unwrap());
                        }
                        Fragment::NonTerminal(..) => {
                            write!(&mut f_print, "\tnonterm {}\n", e.0);
                        }
                        Fragment::Expression(_, _) => {
                            write!(&mut f_print, "\texpression {}\n", e.0);
                        }
                        Fragment::Nop => { write!(&mut f_print, "\tNOP\n"); }
                    }
                }
            }
            Fragment::Terminal(x) => {
                write!(&mut f_print, "Terminal {}: \"{}\"", id.0, std::str::from_utf8(x).unwrap());
            }
            Fragment::NonTerminal(options, ..,  shortest) => {
                write!(&mut f_print, "Non-terminal {}: with {} options and {} shortest paths:\n", id.0, options.len(), shortest.len());
                for e in options {
                    let f_e = self.fragments.get(e.0).unwrap();
                    match f_e {
                        Fragment::Terminal(term) => {
                            write!(&mut f_print, "\tterm \"{}\"\n", std::str::from_utf8(term).unwrap());
                        }
                        Fragment::NonTerminal(..) => {
                            write!(&mut f_print, "\tnonterm {}\n", e.0);
                        }
                        Fragment::Expression(_, _) => {
                            write!(&mut f_print, "\texpression {}\n", e.0);
                        }
                        Fragment::Nop => { write!(&mut f_print, "\tNOP\n"); }
                    }
                }
            }
            Fragment::Nop => { write!(&mut f_print, "NOP\n"); }
        }

        debug!("{}", f_print.clone());
    }

    // Initialize the RNG
    pub fn seed(&self, val: usize) {
        self.seed.set(val);
    }

    // get a random value
    pub fn rand(&self) -> usize {
        let mut seed = self.seed.get();
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 43;

        self.seed.set(seed);
        seed
    }

    /**
        Turns the grammar structure into a Rust program containing many correlated functions.
        Fuzzer instead of interpreting json grammar, turns it into a generation tool.
        It outputs a compiled Rust program able to generate inputs like generate function.
    **/
    pub fn program<P: AsRef<Path>>(&self, path: P, max_depth: usize) {
        let mut program = String::new();
        program += &*format!(r#"
#[allow(unused)]
use std::cell::Cell;
use std::time::Instant;

fn main(){{
    let mut fuzzer = Fuzzer{{
        seed: Cell::new(0x34cc028e11b4f89b),
        buf:   Vec::new(),
    }};

    let mut generated = 0usize;
    let it = Instant::now();
    for iters in 1u64.. {{
        fuzzer.buf.clear();
        fuzzer.fragment_{}(0);
        generated += fuzzer.buf.len();
        // Filter to reduce the amount of times printing occurs
        if (iters & 0xfffff) == 0 {{
            let elapsed = (Instant::now() - it).as_secs_f64();
            let bytes_per_sec = generated as f64 / elapsed;
            print!("MiB/sec: {{:12.4}} | example: {{}}\n", bytes_per_sec / 1024. / 1024., String::from_utf8_lossy(&*fuzzer.buf));
        }}
    }}

}}

struct Fuzzer{{
    seed: Cell<usize>,
    buf:   Vec<u8>,
}}
impl Fuzzer {{
    fn rand(&self) -> usize {{
        let mut seed = self.seed.get();
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 43;

        self.seed.set(seed);
        seed
    }}
"#, self.name_to_fragment.get(&*self.start).unwrap().0);
        // create a function for each fragment in grammar
        for (id, fragment) in self.fragments.iter().enumerate() {
            program += &format!("   fn fragment_{}(&mut self, depth: usize){{\n", id);
            match fragment {
                Fragment::NonTerminal(options, probs, .., shortest) => {
                    // if only one shortest, there is no need to put it in a vector
                    let one_shortest = shortest.len() == 1;

                    // if depth not exceeded
                    program += &format!("       if depth < {}{{\n", max_depth);
                    // get a random number using distributed probability
                    program += &format!("           //options: {}, prob: {} \n", options.len(), probs.len());
                    program += &format!("           match self.rand() % {} {{ \n", probs.len());
                    for (p_id, p) in probs.iter().enumerate() {
                        program += &format!("               {} => self.fragment_{}(depth+1),\n", p_id, p.0);
                    }
                    program += &format!("               _ => unreachable!(),\n");
                    program += "            }\n";
                    // if max_depth exceeded - follow shortest path
                    program += "        } else{\n";
                    // get a random number using distributed probability
                    program += &format!("           //shortest: {} \n", shortest.len());
                    program += &format!("           match self.rand() % {} {{ \n", shortest.len());
                    for (p_id, p) in shortest.iter().enumerate() {
                        program += &format!("               {} => self.fragment_{}(depth+1),\n", p_id, p.0);
                    }
                    program += &format!("               _ => unreachable!(),\n");
                    program += "            }\n";
                    program += "        }\n";


                }
                Fragment::Expression(terms, _) => {
                    // call the function of each term
                    for term in terms{
                        program += &format!("       self.fragment_{}(depth+1);\n", term.0)
                    }
                }
                Fragment::Terminal(value) => {
                    // add the value to the input
                    program += &format!("       self.buf.extend_from_slice(&{:?});\n", value);
                }
                Fragment::Nop => {

                }
            }
            program += "    }\n";
        }
        program += "}\n";
        // print!("{}\n", program);
        // Write out the test application
        std::fs::write(path, program)
            .expect("Failed to create output Rust application");
    }

    /**
        Interprets the grammar structure to generate inputs
    **/
    pub fn generate(&self, stack: &mut Vec<FragmentId>, buf: &mut Vec<u8>, max_depth: i32) {
        // get access to the start node
        let start = self.name_to_fragment.get(&*self.start).unwrap();

        // keep track of expansion depth
        let mut depth = 0;

        // start off working on start
        stack.clear();
        stack.push(*start);

        while !stack.is_empty() {
            let cur = stack.pop().unwrap();

            if max_depth == -1 || depth < max_depth {
                match self.fragments.get(cur.0).unwrap() {
                    Fragment::NonTerminal(_, probs, ..) => {
                        let sel = probs[self.rand() % probs.len()];
                        stack.push(sel);
                        depth += 1;
                    }
                    Fragment::Expression(expr, _) => {
                        // we must process all of these in sequence
                        // take expr slice and append all elements to stack vec
                        expr.iter().rev().for_each(|&x| stack.push(x));
                        depth += 1;
                    }
                    Fragment::Terminal(value) => {
                        buf.extend_from_slice(&*value);
                        if buf.len() > 16 * 1024 {
                            break;
                        }
                    }
                    _ => {}
                }
            } else {
                match self.fragments.get(cur.0).unwrap() {
                    Fragment::Expression(terms, _) => {
                        terms.iter().rev().for_each(|&x| stack.push(x));
                    }
                    Fragment::NonTerminal(.., mins) => {
                        let sel = if mins.len() == 1 { mins[0] } else { mins[self.rand() % mins.len()] };
                        stack.push(sel);
                    }
                    Fragment::Terminal(value) => {
                        buf.extend_from_slice(value);
                    }
                    _ => {}
                }
            }
        }
    }
}


fn main() -> std::io::Result<()> {
    env_logger::init();
    info!("starting up");
    // serialize grammar input
    let grammar_json: GrammarJson = serde_json::from_slice(&std::fs::read("res/expressions.json")?)?;
    let grammar = Grammar::new(&grammar_json);
    debug!("{:?}", grammar.name_to_fragment);

    grammar.program("src/F2.rs", 20);

    // let mut buf: Vec<u8> = Vec::new();
    //
    // let mut stack = Vec::new();
    // let mut rng = rand::thread_rng();
    // grammar.seed(rng.gen::<i8>() as usize);
    //
    // let mut generated = 0usize;
    // let it = Instant::now();
    //
    // for iters in 1u64.. {
    //     buf.clear();
    //     grammar.generate(&mut stack, &mut buf, 64);
    //     // debug!("{}", String::from_utf8_lossy(&buf));
    //     // num of u8 = bytes bc byte = 8 bits
    //     generated += buf.len();
    //
    //     if (iters & 0xffff) == 0 {
    //         let elapsed = (Instant::now() - it).as_secs_f64();
    //         let bytes_per_sec = generated as f64 / elapsed;
    //         print!("MiB sec: {:12.6} | Example: {:#?}\n", bytes_per_sec / 1024. / 1024., String::from_utf8_lossy(&buf));
    //     }
    // }

    Ok(())
}
