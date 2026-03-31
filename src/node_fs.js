import { openSync, readSync, closeSync, statSync } from "node:fs";
import { cwd } from "node:process";

export function readFileRangeSync(filePath, offset, length) {
  const fd = openSync(filePath, "r");
  try {
    const buf = Buffer.alloc(length);
    const bytesRead = readSync(fd, buf, 0, length, offset);
    return new Uint8Array(buf.buffer, buf.byteOffset, bytesRead);
  } finally {
    closeSync(fd);
  }
}

export function fileSizeSync(filePath) {
  return statSync(filePath).size;
}

export function getCwd() {
  return cwd();
}
