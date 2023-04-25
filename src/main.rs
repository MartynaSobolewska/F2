mod F2;

#[macro_use]
extern crate log;
extern crate core;

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
// use std::collections::btree_map::BTreeMap;
// use std::collections::btree_set::BTreeSet;
// use std::collections::vec_deque::VecDeque;
use serde::{Deserialize, Serialize};
use log::{info, warn};
use std::fmt::Write;
use std::path::Path;
use std::process::Command;
use std::string::String;
use std::time::Instant;
use rand::Rng;

/// If this is `true` then the output file we generate will not emit any
/// unsafe code. I'm not aware of any bugs with the unsafe code that I use and
/// thus this is by default set to `false`. Feel free to set it to `true` if
/// you are concerned. - Gamozolabs (https://github.com/gamozolabs/fzero_fuzzer/blob/master)
const SAFE_ONLY: bool = true;

/// If this is true, the fuzzer will never stop unwrapping elements once the max_depth is reached,
/// leading to full completeness of the outputs but lower throughput
const FULL_CORRECTNESS: bool = true;

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
        for (f_name, _) in grammar_json.0.iter() {
            // if already in grammar, throw an error
            if grammar.name_to_fragment.contains_key(f_name) {
                panic!("{} fragment declaration repeats in grammar.", f_name);
            }
            // else - allocate a new empty fragment
            let fragment_id = grammar.allocate_fragment(
                Fragment::NonTerminal(
                    Vec::new(), Vec::new(), 0, Vec::new()));

            // save fname-id relation in the map
            grammar.name_to_fragment.insert(
                f_name.clone(),
                fragment_id,
            );
        }

        //check if grammar contains the start token
        if !grammar.name_to_fragment.contains_key(&*grammar.start) {
            panic!("Grammar does not contain the \"<start>\" fragment.")
        }

        for (f_name, options) in grammar_json.0.iter() {
            // get non-term id
            let f_id = grammar.name_to_fragment[f_name];

            // store all the options for non-terminal
            let mut expressions = Vec::new();
            // keep track of probabilities for different options
            let mut probabilities = Vec::new();
            // a flag to indicate if some option has a non-default probability
            let mut some_probabilities = false;

            // iterate through options (expressions)
            for option in options {
                // all terms (chronological in current expression)
                let mut terms = Vec::new();

                // check if there is a probability
                //probability of current expression being chosen
                let mut probability: Option<usize> = None;

                let mut option_without_prob = option.clone();

                // check if current fragment has a probability
                match option.last() {
                    Some(str) => {
                        if str.starts_with("p=") {
                            let num = String::from(&str[2..]).parse::<u32>();
                            match num {
                                Ok(ok) => {
                                    probability = Some(ok as usize);
                                    // get rid of probability field
                                    option_without_prob.pop();
                                    some_probabilities = true;
                                }
                                Err(_) => {
                                    panic!(
                                        "A fragment {} contains an option with badly formatted probability field!"
                                        , f_name
                                    );
                                }
                            }
                        }
                    }
                    None => {
                        // no children
                    }
                }

                //default probability - 1
                probability = if probability.is_some() { probability } else { Some(1) };
                // probability should never be 0
                if probability == Some(0) {
                    panic!("Fragment {} contains a child with a zero probability!", f_name);
                }

                probabilities.push(probability.unwrap());

                for term in &option_without_prob {
                    let mut term_id;
                    // non-terminal
                    if grammar.name_to_fragment.contains_key(term) {
                        term_id = grammar.name_to_fragment[term];
                    } else {
                        // create a new terminal fragment
                        term_id = grammar.allocate_fragment(
                            Fragment::Terminal(term.as_bytes().to_vec()));
                    }
                    terms.push(term_id);
                }

                // allocate a new fragment with terms
                expressions.push(grammar.allocate_fragment(
                    Fragment::Expression(terms, u32::MAX)));
            }

            // change the contents of currently processed non-terminal
            let non_term = &mut grammar.fragments[f_id.0];

            // if some non-zero probabilities present, add the term that number of times
            let mut expressions_with_probabilities = Vec::new();
            if some_probabilities {
                for (index, p) in probabilities.iter().enumerate() {
                    for _ in 0..*p {
                        expressions_with_probabilities.push(expressions[index]);
                    }
                }
            } else {
                expressions_with_probabilities = expressions.clone();
            }

            *non_term =
                Fragment::NonTerminal(expressions,
                                      expressions_with_probabilities, u32::MAX,
                                      Vec::new());

        }
        grammar
    }

    /**
        finds the vector of next fragments to unwrap in order to achieve the shortest path
    **/
    fn find_steps(&mut self) {
        // think of the grammar as a connected graph, traverse each layer
        // depth first search to assign steps to terminal for each fragment

        debug!("*******************************************************************");
        debug!("*                               STEPS                             *");
        debug!("*******************************************************************\n");

        // initially stack only contains the start node
        let mut stack: VecDeque<FragmentId> = VecDeque::new();
        stack.push_back(*self.name_to_fragment.get(&self.start).unwrap());

        // nodes that pushed children onto stack
        let mut visited: HashSet<usize> = HashSet::new();
        // nodes pushed back because of a cycle
        let mut pushed: HashSet<usize> = HashSet::new();

        // keep track of number of inconclusive fragments for debug purposes
        let mut inconclusive = 0;

        // traverse entire graph
        while !stack.is_empty() {
            // get currently processed fragment
            let curr_id = stack.back().unwrap().0;
            let curr = self.fragments.get(curr_id).unwrap();

            // if it has been visited, check if you can find its shortest or if it is in a cycle
            if visited.contains(&curr_id) {
                stack.pop_back();
                match curr {
                    // if non-term, min+1 = min_steps
                    // all the fragments with steps min get pushed into shortest vector
                    Fragment::NonTerminal(options, probabilities, ..) => {
                        let mut min = u32::MAX;
                        let mut shortest = Vec::new();

                        // find the options with least amount of steps
                        for o in options {
                            // get the fragment
                            let option = self.fragments.get(o.0).unwrap();
                            match option {
                                // terminal has steps = 1
                                Fragment::Terminal(_) => {
                                    // if new min, clear the shortest vector and add new shortest
                                    if min > 1 {
                                        min = 1;
                                        shortest.clear();
                                    }
                                    // if min is 1, add option's id to shortest
                                    if min == 1 {
                                        shortest.push(*o);
                                    }
                                }
                                // non-terminal has already computed steps (s),
                                // if s = u32::MAX, it is in a loop
                                // so it will never be the shortest option
                                Fragment::NonTerminal(_, _, s, ..) => {
                                    if (*s) < min {
                                        min = *s + 1;
                                        shortest.clear();
                                        shortest.push(*o);
                                    } else if (*s) == min {
                                        shortest.push(*o);
                                    }
                                }
                                // Expression similarly to non-term has precomputed steps s
                                Fragment::Expression(_, s) => {
                                    if (*s) < min {
                                        min = *s + 1;
                                        shortest.clear();
                                        shortest.push(*o);
                                    } else if (*s) == min {
                                        shortest.push(*o);
                                    }
                                }
                                // Nop has steps 0, always shortest path
                                Fragment::Nop => {
                                    if min > 0 {
                                        min = 0;
                                        shortest.clear();
                                    }

                                    shortest.push(*o);
                                }
                            }
                        }
                        // if all children have cycles, push back,
                        // try to compute again once again later
                        if min == u32::MAX {
                            // if it has already been pushed back before- inconclusive cycle
                            // can't calculate min steps, else try one more time, add it to pushed.
                            if !pushed.contains(&curr_id) {
                                stack.push_front(FragmentId(curr_id));
                                pushed.insert(curr_id);
                            } else {
                                debug!("No way to calculate the shortest for non-terminal: {}", curr_id);
                                inconclusive += 1;
                                // no way to calculate shortest so all options are
                                self.fragments[curr_id] =
                                    Fragment::NonTerminal(options.clone(),
                                                          probabilities.clone(),
                                                          min,
                                                          options.clone())
                            }
                        }
                        // if min is not u32::MAX, shortest have been found.
                        else {
                            if shortest.len() == 0 {
                                self.print_fragment(FragmentId(curr_id));
                                panic!("HERE");
                            }
                            self.fragments[curr_id] =
                                Fragment::NonTerminal(options.clone(),
                                                      probabilities.clone(),
                                                      min,
                                                      shortest.clone())
                        }
                    }
                    // for expression shortest path is sum of all children's shortest
                    // so if any of the children is inconclusive, so will be the expression
                    Fragment::Expression(terms, mut _steps) => {
                        let mut min = 0;
                        // if a child is dependent on a parent cannot calculate minimum
                        let mut uncalculated_child = false;
                        for t in terms {
                            let term = self.fragments.get(t.0).unwrap();
                            // calculate the steps
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

                        // if one of the children is inconclusive,
                        // try to compute expression again, later by pushing it to the end
                        if uncalculated_child {
                            min = u32::MAX;
                            if !pushed.contains(&curr_id) {
                                stack.push_front(FragmentId(curr_id));
                                pushed.insert(curr_id);
                            } else {
                                // can't calculate min steps
                                debug!("Can't calculate min_steps for expression {}", curr_id);
                                inconclusive += 1;
                                _steps = min;
                            }
                        } else {
                            _steps = min;
                        }
                    }
                    _ => {
                        warn!("encountered a Term while assigning steps.")
                    }
                }
            }
            // if current has not been visited yet, push all non-term and expression children
            // onto the stack to be processed first
            else {
                // if non-term or expression, push unvisited children
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


        debug!("Found shortest paths for all possible fragments. \
        {} fragments are in an inconclusive cycle.", inconclusive);
        if inconclusive != 0 {
            println!("Couldn't find shortest path for {} out of {} fragments. \
            If the number is too large, consider rewriting a less recursive grammar.",
                     inconclusive, self.fragments.len());
        }
        if inconclusive == self.fragments.len() {
            panic!("Couldn't find shortest path for any fragment. Bad grammar formatting.");
        }
        println!();
    }

    fn optimise(&mut self) {
        debug!("*******************************************************************");
        debug!("*                        OPTIMISE GRAMMAR                         *");
        debug!("*******************************************************************\n");

        // after the struct has been changed, there is maybe something left to optimise
        // that was not detected before
        let mut optimised = true;
        let mut single_term_nonterms = 0;
        let mut single_term_expressions = 0;
        let mut NOPs = 0;

        // Keeps track of fragment identifiers which resolve to nops
        let mut nop_fragments = BTreeSet::new();
        while optimised {
            optimised = false;

            // Go through each fragment, looking for potential optimizations
            for f_id in 0..self.fragments.len() {
                match self.fragments[f_id].clone() {
                    Fragment::NonTerminal(options, ..) => {
                        // if only one option, no need for non-term, replace it with the option
                        if options.len() == 1 {
                            self.fragments[f_id] = self.fragments[options[0].0].clone();
                            optimised = true;
                            single_term_nonterms += 1;
                        }
                    }
                    Fragment::Expression(expr, ..) => {
                        // if no items in expression, turn it into a NOP
                        if expr.len() == 0 {
                            self.fragments[f_id] = Fragment::Nop;
                            optimised = true;
                            nop_fragments.insert(f_id);
                            NOPs += 1;
                        }

                        // if only one term in expression, replace it with the term
                        if expr.len() == 1 {
                            self.fragments[f_id] = self.fragments[expr[0].0].clone();
                            optimised = true;
                            single_term_expressions += 1;
                        }


                        if let Fragment::Expression(exprs, ..) =
                        &mut self.fragments[f_id] {
                            // Only retain fragments which are not nops
                            exprs.retain(|x| {
                                if nop_fragments.contains(&x.0) {
                                    // Fragment was a nop, remove it
                                    optimised = true;
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
        println!("Number of single-term expressions simplified: {}", single_term_expressions);
        println!("Number of single-term non-terminals simplified: {}", single_term_nonterms);
        println!("Number of NOPs found: {}", NOPs);
        println!();
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
        debug!("*******************************************************************");
        debug!("*                        OPTIMISE STEPCOUNT                       *");
        debug!("*******************************************************************\n");

        // initially, start node
        // stack to store fragments
        let mut stack: VecDeque<FragmentId> = VecDeque::new();
        stack.push_back(*self.name_to_fragment.get(&self.start).unwrap());
        // store fragments that have been in the stack
        let mut visited: HashSet<usize> = HashSet::new();
        // store fragments that have been in a chain before
        let mut chained: HashSet<usize> = HashSet::new();
        let mut visited_nonterms = 0;

        // keep track of a total number of skipped nodes (for data)
        let mut total_skipped = 0;

        // keep track of a number of paths that have been shortened (for data)
        let mut no_of_chains = 0;

        while !stack.is_empty() {
            // discover fragments with single shortest path
            // pop fragment from top of the stack
            let parent_id = stack.pop_back().unwrap().0;
            let parent = self.fragments.get(parent_id).unwrap();
            match parent {
                Fragment::NonTerminal(options, .., shortest) => {
                    visited_nonterms += 1;

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
                        // search for non-terminals to skip
                        // add all of single path non-terminal chain to the vector
                        let mut chain: Vec<usize> = Vec::new();
                        let mut curr = parent_id;
                        let mut chain_broke = false;
                        while !chain_broke {
                            chain.push(curr.clone());
                            // get next in chain
                            if let Fragment::NonTerminal(.., prev_shortest) =
                            self.fragments.get(curr.clone()).unwrap().clone() {
                                curr = prev_shortest[0].0
                            }

                            let curr_fragment = self.fragments.get(curr.clone()).unwrap();
                            match curr_fragment {
                                // if next in chain is a non-terminal
                                Fragment::NonTerminal(.., curr_shortest) => {
                                    if curr_shortest.len() != 1 {
                                        chain_broke = true;
                                    }
                                    if !visited.contains(&curr) {
                                        // add the non-term to the stack for further exploration
                                        stack.push_back(FragmentId(curr));
                                    }
                                }
                                _ => {
                                    chain_broke = true;
                                }
                            }
                        }
                        if chain.len() > 1 {
                            no_of_chains += 1;
                            // kipping all in chain except for last one
                            total_skipped += chain.len() - 1;
                            let new_shortest = Vec::from([FragmentId(curr)]);
                            // change shortest path of each fragment in the chain
                            while !chain.is_empty() {
                                let curr_id = chain.pop().unwrap();
                                chained.insert(curr_id);
                                let f = self.fragments.get(curr_id).unwrap();
                                match f {
                                    Fragment::NonTerminal(opt, probs, steps, _) => {
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
        println!("Skipped a total of {} fragments in {} shortest paths", total_skipped, no_of_chains);
        println!();
    }

    fn allocate_fragment(&mut self, fragment: Fragment) -> FragmentId {
        let id = self.fragments.len();
        self.fragments.push(fragment);
        FragmentId(id)
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
            Fragment::NonTerminal(options, .., shortest) => {
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
#![allow(unused)]
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
        if (iters & 0xffffff) == 0 {{
            let elapsed = (Instant::now() - it).as_secs_f64();
            let bytes_per_sec = generated as f64 / elapsed;
            print!("MiB/sec: {{:12.4}} | example: {{}}\n", bytes_per_sec / 1024. / 1024., String::from_utf8_lossy(&*fuzzer.buf));
            // print!("MiB/sec: {{:12.4}}\n", bytes_per_sec / 1024. / 1024.);
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
            match fragment {
                Fragment::NonTerminal(options, probs, .., shortest) => {
                    program += &format!("   fn fragment_{}(&mut self, mut depth: usize){{\n", id);
                    program += &format!("       if depth > {}{{\n", max_depth-1);
                    if !FULL_CORRECTNESS{
                        program += &format!("           return;\n");
                    }else{
                        // get a random number using distributed probability
                        program += &format!("           //shortest: {} \n", shortest.len());
                        program += &format!("           match self.rand() % {} {{ \n", shortest.len());
                        for (p_id, p) in shortest.iter().enumerate() {
                            match self.fragments.get(p.0).unwrap() {
                                Fragment::Terminal(value) => {
                                    // open the arm
                                    program += &format!("               {} => {{\n", p_id);

                                    if SAFE_ONLY {
                                        program += &format!("                   self.buf.extend_from_slice(&{:?})\n", value);
                                    }else {
                                        program += &format!(r#"                   {{
                    unsafe {{
                        let old_size = self.buf.len();
                        let new_size = old_size + {};
                        if new_size > self.buf.capacity() {{
                            self.buf.reserve(new_size - old_size);
                        }}
                        std::ptr::copy_nonoverlapping({:?}.as_ptr(), self.buf.as_mut_ptr().offset(old_size as isize), {});
                        self.buf.set_len(new_size);
                    }}
"#, value.len(), value, value.len());
                                    }
                                    // close the arm
                                    program += &format!("               }},\n");
                                }
                                Fragment::Nop => {
                                    program += &format!("               {} => {{}},\n", p_id);
                                }
                                _ => program += &format!("               {} => self.fragment_{}(depth+1),\n", p_id, p.0)
                            }
                        }
                        program += &format!("               _ => unreachable!(),\n");
                        program += "            }\n";
                    }
                    // depth < max_depth
                    program += "        } else{\n";
                    // get a random number using distributed probability
                    program += &format!("           match self.rand() % {} {{ \n", probs.len());
                    for (p_id, p) in probs.iter().enumerate() {
                        match self.fragments.get(p.0).unwrap() {
                            Fragment::Nop => program += &format!("               {} => {{}},\n", p_id),
                            Fragment::Terminal(value) => {
                                if SAFE_ONLY {
                                    program += &format!("               {} => self.buf.extend_from_slice(&{:?}),\n", p_id, value);
                                }else {
                                    program += &format!(r#"               {} => {{
                    unsafe {{
                        let old_size = self.buf.len();
                        let new_size = old_size + {};
                        if new_size > self.buf.capacity() {{
                            self.buf.reserve(new_size - old_size);
                        }}
                        std::ptr::copy_nonoverlapping({:?}.as_ptr(), self.buf.as_mut_ptr().offset(old_size as isize), {});
                        self.buf.set_len(new_size);
                    }}
                }}
"#, p_id, value.len(), value, value.len());
                                }
                            }
                            _ => program += &format!("               {} => self.fragment_{}(depth+1),\n", p_id, p.0)
                        }
                    }
                    program += &format!("               _ => unreachable!(),\n");
                    program += "            }\n";
                    program += "        }\n";
                    program += "    }\n";
                }
                Fragment::Expression(terms, _) => {
                    program += &format!("   fn fragment_{}(&mut self, depth: usize){{\n", id);
                    if !FULL_CORRECTNESS{
                        program += &format!("       if depth > {}{{\n", max_depth-1);
                        program += &format!("           return;\n       }}\n");
                    }
                    // call the function of each term
                    for term in terms {
                        match self.fragments.get(term.0).unwrap() {
                            Fragment::NonTerminal(..) | Fragment::Expression(..) => {
                                program += &format!("       self.fragment_{}(depth+1);\n", term.0);
                            }
                            Fragment::Terminal(value) => {
                                if SAFE_ONLY {
                                    program += &format!("       self.buf.extend_from_slice(&{:?});\n", value);
                                } else {
                                    // following gamozolabs observaion on extend_from_slice,
                                    // this unsafe snippet does the same thing causing speedup
                                    program += &format!(r#"
            unsafe {{
                let old_size = self.buf.len();
                let new_size = old_size + {};
                if new_size > self.buf.capacity() {{
                    self.buf.reserve(new_size - old_size);
                }}
                std::ptr::copy_nonoverlapping({:?}.as_ptr(), self.buf.as_mut_ptr().offset(old_size as isize), {});
                self.buf.set_len(new_size);
            }}
    "#, value.len(), value, value.len());
                                }
                            }
                            _ => {}
                        }
                    }
                    program += "    }\n";
                }
                Fragment::Terminal(_) => {
                    // no need for a function, inlined
                }
                Fragment::Nop => {}
            }
        }
        program += "}\n";
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
            // println!("{:#?}\n", String::from_utf8_lossy(&buf));

            let cur = stack.pop().unwrap();

            if buf.len() > 128 * 128 * 2 {
                return;
            }

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
    // Get access to the command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        print!("usage: f2 <grammar json> <output Rust file> <output binary name> <max depth>\n");
        return Ok(());
    }

    // Read the grammar json
    let grammar: GrammarJson = serde_json::from_slice(
        &std::fs::read(&args[1])?)?;
    println!("Loaded grammar json\n");

    // Convert the grammar file to the Rust structure
    let mut grammar = Grammar::new(&grammar);
    println!("Converted grammar to a Rust structure\n");

    // turn all 1-child expressions into the child itself, add NOPs where it is possible
    grammar.optimise();
    println!("Optimised grammar\n");

    // find the shortest paths to completion
    grammar.find_steps();
    println!("Found shortest completion paths in grammar.");

    // skip nodes in the paths if possible
    grammar.optimise_stepcount();

    // Generate a Rust application
    grammar.program(&args[2],
                    args[4].parse().expect("Invalid digit in max depth"));
    print!("Generated Rust source file\n");

    // Compile the application
    // rustc -O -g test.rs -C target-cpu=native
    let status = Command::new("rustc")
        .arg("-O")                // Optimize the binary
        .arg("-g")                // Generate debug information
        .arg(&args[2])            // Name of the input Rust file
        .arg("-C")                // Optimize for the current microarchitecture
        .arg("target-cpu=native")
        .arg("-o")                // Output filename
        .arg(&args[3]).spawn()?.wait()?;
    assert!(status.success(), "Failed to compile Rust binary");
    print!("Created Rust binary!\n");

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
    //     grammar.generate(&mut stack, &mut buf, );
    //     // debug!("{}", String::from_utf8_lossy(&buf));
    //     // num of u8 = bytes bc byte = 8 bits
    //     generated += buf.len();
    //
    //     if (iters & 0xffffff) == 0 {
    //         let elapsed = (Instant::now() - it).as_secs_f64();
    //         let bytes_per_sec = generated as f64 / elapsed;
    //         print!("MiB sec: {:12.6} | Example: {:#?}\n", bytes_per_sec / 1024. / 1024., String::from_utf8_lossy(&buf));
    //     }
    // }

    Ok(())
}
