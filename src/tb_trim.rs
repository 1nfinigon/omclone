use std::fs::{self, File};
use std::mem;
use std::io::{self, BufReader, BufWriter, Read, Write};

use eyre::Result;
use crate::utils;

/// Returns Ok(buf, is_eof)
fn read_exact<'a, R: Read + ?Sized>(r: &mut R, mut buf: &'a mut [u8]) -> Result<(&'a [u8], bool)> {
    let mut offset = 0;
    while !buf[offset..].is_empty() {
        match r.read(&mut buf[offset..]) {
            Ok(0) => {
                return Ok((&buf[..offset], true));
            },
            Ok(n) => {
                offset += n;
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok((buf, false))
}

pub fn main() -> Result<()> {
    let mut paths = Vec::new();
    utils::read_file_suffix_recurse(&mut |p| {
        if p.file_name().and_then(|f| f.to_str()).is_some_and(|f| f.starts_with("events.out")) {
            paths.push(p);
        }
    }, "", "test/tensorboard");

    for p in paths {
        println!("processing {:?}", p);

        let new_p = p.with_file_name("tmp");

        let mut r = BufReader::new(File::open(&p)?);
        let mut w = BufWriter::new(File::create(&new_p)?);

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
                let (data_buffer, is_eof) = read_exact(&mut r, &mut data_buffer[..length + 8])?;
                w.write_all(data_buffer)?;
                if is_eof {
                    println!("file was truncated early (read {}, expected {})", data_buffer.len(), length + 8);
                    break;
                }
            } else {
                r.seek_relative((length + 8).try_into().unwrap())?;
            }
        }

        mem::drop(r);
        mem::drop(w);

        fs::remove_file(&p)?;
        fs::rename(&new_p, &p)?;
    }

    Ok(())
}
