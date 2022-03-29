use pyo3::Python;
use pyo3::prelude::*;
use std::collections::HashMap;

/*
enum NestedList {
    List(Vec<NestedList>),
    Val(f32)
}

fn to_native_list(py: &Python, obj: &PyAny) -> PyResult<NestedList> {
    if obj.is_instance_of::<PyList>().unwrap() {
        let items: Vec<&PyAny> = obj.extract()?;
        let native = items.iter().map(|v| to_native_list(&py, v)).collect::<PyResult<Vec<NestedList>>>()?;
        return Ok(NestedList::List(native));
    }

    Ok(NestedList::Val(obj.extract()?))
}
*/

type Residues = Vec<Vec<f32>>;

#[pyfunction]
fn pad_flatten(pdb_data: HashMap<String, &PyAny>, longest_num_residues: usize, longest_residue_length: usize) -> HashMap<String, Vec<f32>> {
    pdb_data.into_iter()
        .map(|(k, v)| (k, v.extract::<Residues>().unwrap()))
        .map(|(k, v)| {
            let mut v: Vec<_> = v.iter()
                .map(|s| {
                    let mut s = s.to_vec();
                    let long = vec![0 as f32; longest_residue_length - s.len()];
                    s.extend(long);
                    s
                }
                ).collect();
            v.resize(longest_num_residues, vec![0 as f32; longest_residue_length]);
            (k, v.into_iter().flatten().collect())
        }).collect()
}

/*
fn pad(residues: PyResult<Residues>, longest_residue_length: u32) -> PyResult<Residues> {
    Ok(Vec::new())
}
*/

#[pymodule]
fn rpdb(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pad_flatten, m)?)?;
    Ok(())
}