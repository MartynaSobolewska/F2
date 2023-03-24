#[macro_use]
extern crate log;
extern crate core;

use std::borrow::Borrow;
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ops::Add;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use log::{info, warn};
use std::fmt::Write;
use std::string::String;
use rand::Rng;

// Json representation of the data struct
// Map Fragment name : List<List <Fragment Names>>
#[derive(Serialize, Deserialize, Debug, Default)]
struct GrammarJson(HashMap<String, Vec<Vec<String>>>);

#[derive(Clone, Debug, Copy, Default, PartialEq)]
struct FragmentId(usize);

#[derive(Clone, Debug)]
enum Fragment {
    // nonterminal contains a vector of fragments (some might be non-terminal) and probabilities
    NonTerminal(Vec<FragmentId>, Vec<u32>, u32, Vec<FragmentId>),
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

            // for each option - wrapped as expression due to json formatting
            for mut expression in f_options {
                let mut probability: Option<u32> = None;

                // check if it has a probability
                match expression.last() {
                    Some(str) => {
                        if str.starts_with("p=") {
                            let num = String::from(&str[2..]).parse::<u32>();
                            match num {
                                Ok(ok) => { probability = Some(ok); }
                                Err(e) => {
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
                                subfragment_id = (*grammar.name_to_fragment.get(subfragment).unwrap());
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
            // reallocate the updated version of the non-terminal with body onto the vector
            grammar.reallocate_nonterm(
                Fragment::NonTerminal(options, probabilities, u32::MAX, Vec::new()),
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
                    Fragment::NonTerminal(options, probabilities, steps, mins) => {
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
                                Fragment::NonTerminal(_, _, s, _) => {
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
                    Fragment::Expression(terms, mut steps) => {
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
                                steps = min;
                            }
                        } else {
                            steps = min;
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
                    Fragment::NonTerminal(fragments, _, _, _) => {
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
        // track if optimisation had an effect
        let mut optimise = true;
        while optimise {
            optimise = false;
            for f_id in 0..self.fragments.len() {
                match self.fragments[f_id].clone() {
                    Fragment::NonTerminal(options, _, _, _) => {
                        if options.len() == 1 {
                            self.fragments[f_id] = self.fragments[options[0].0].clone();
                            optimise = true;
                            debug!("Non-terminal {} with one option changed into:", f_id);
                            self.print_fragment(FragmentId(f_id));
                        }
                    }
                    Fragment::Expression(mut terms, _) => {
                        if terms.len() == 0 {
                            self.fragments[f_id] = Fragment::Nop;
                            optimise = true;
                            debug!("Expression {} with 0 fragments changed into NOP", f_id);
                        } else if terms.len() == 1 {
                            self.fragments[f_id] = self.fragments[terms[0].0].clone();
                            optimise = true;
                            debug!("Expression {} with 1 fragment changed into:", f_id);
                            self.print_fragment(FragmentId(f_id));
                        } else {
                            let mut new_terms = Vec::new();
                            for term in terms {
                                match self.fragments[term.0] {
                                    Fragment::Nop => {
                                        debug!("Nop deleted from expression {}", f_id);
                                    }
                                    _ => { new_terms.push(term) }
                                }
                            }
                            terms = new_terms;
                        }
                    }
                    Fragment::Nop | Fragment::Terminal(_) => {
                        // nothing to optimise
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
                Fragment::NonTerminal(options, _, _, shortest) => {
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
        let mut id = FragmentId(0);
        if self.name_to_fragment.contains_key(f_name) {
            id = *self.name_to_fragment.get(f_name).unwrap();
            match self.fragments.get(id.0).unwrap() {
                Fragment::NonTerminal(_, _, _, _) => {
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
                        Fragment::NonTerminal(_, _, _, _) => {
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
            Fragment::NonTerminal(options, _, _, shortest) => {
                write!(&mut f_print, "Non-terminal {}: with {} options and {} shortest paths:\n", id.0, options.len(), shortest.len());
                for e in options {
                    let f_e = self.fragments.get(e.0).unwrap();
                    match f_e {
                        Fragment::Terminal(term) => {
                            write!(&mut f_print, "\tterm \"{}\"\n", std::str::from_utf8(term).unwrap());
                        }
                        Fragment::NonTerminal(_, _, _, _) => {
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
                    Fragment::NonTerminal(options, _, _, _) => {
                        let sel = options[self.rand() % options.len()];
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
                    Fragment::NonTerminal(_, _, _, mins) => {
                        let sel = if mins.len() == 0 { mins[0] } else { mins[self.rand() % mins.len()] };
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
    let grammar_json: GrammarJson = serde_json::from_slice(&std::fs::read("res/test.json")?)?;
    let mut grammar = Grammar::new(&grammar_json);
    debug!("{:?}", grammar.name_to_fragment);


    let mut buf: Vec<u8> = Vec::new();

    let mut stack = Vec::new();
    let mut rng = rand::thread_rng();
    grammar.seed(rng.gen::<i8>() as usize);

    let mut generated = 0usize;
    let it = Instant::now();

    for iters in 1u64.. {
        buf.clear();
        grammar.generate(&mut stack, &mut buf, 64);
        // num of u8 = bytes bc byte = 8 bits
        generated += buf.len();

        if (iters & 0xfffff) == 0 {
            let elapsed = (Instant::now() - it).as_secs_f64();
            let bytes_per_sec = generated as f64 / elapsed;
            print!("MiB sec: {:12.6} | Example: {:#?}\n", bytes_per_sec / 1024. / 1024., String::from_utf8_lossy(&buf));
        }
    }

    Ok(())
}
