use std::fs::{self, File};
use std::mem;
use std::io::{self, BufReader, BufWriter, Read, Write};

use eyre::Result;
use crate::utils;

pub fn main() -> Result<()> {
    let mut paths = Vec::new();
    utils::read_file_suffix_recurse(&mut |p| {
        if p.file_name().and_then(|f| f.to_str()).is_some_and(|f| f.starts_with("events.out")) {
            paths.push(p);
        }
    }, "", "test/tensorboard");

    for p in paths {
        println!("processing {:?}", p);

        let old_p = p.with_file_name("tmp");
        std::fs::rename(&p, &old_p)?;

        let mut r = BufReader::new(File::open(&old_p)?);
        let mut w = BufWriter::new(File::create_new(&p)?);

        loop {
            let mut length_buffer = [0u8; 8];
            match r.read_exact(&mut length_buffer) {
                Ok(()) => (),
                Err(e) =>
                    if e.kind() == io::ErrorKind::UnexpectedEof { break; }
                    else {
                        return Err(e.into());
                    },
            }
            let length = u64::from_le_bytes(length_buffer) as usize;
            if length < 16384 {
                w.write_all(&length_buffer)?;

                let mut data_buffer = [0u8; 16384 + 8];
                r.read_exact(&mut data_buffer[..length + 8])?;
                w.write_all(&data_buffer[..length + 8])?;
            } else {
                r.seek_relative((length + 8).try_into().unwrap())?;
            }
        }

        mem::drop(r);
        mem::drop(w);

        fs::remove_file(old_p)?;
    }

    Ok(())
}
