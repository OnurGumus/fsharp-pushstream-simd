namespace PushStream6.SimdBenchmark

open System
open System.Numerics
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Running

// ---------------------------------------------------------------------------
// A SIMD push stream (specialized to int for clarity).
//
// The scalar PushStream's receiver is ('T -> bool): one element at a time, so
// nothing downstream can vectorize. Here the *source owns the loop*, so it can
// push Vector<int> BLOCKS to a vector-receiver and the leftover elements to a
// scalar-receiver. InlineIfLambda fuses the per-block lambda away exactly like
// the scalar version, so the whole pipeline collapses to a vector loop + tail.
//
// Shape: a stream is a function fed two receivers (vector, scalar).
//   No `bool` early-exit here: these are drain/reduce pipelines (sum/max/...),
//   which is the natural fit for SIMD (early-exit per element defeats packing).
// ---------------------------------------------------------------------------
module SimdStream =

  type S = (Vector<int> -> unit) -> (int -> unit) -> unit

  // Source: stream an int[] as full Vector<int> blocks + a scalar remainder.
  let inline ofArray (xs : int[]) : S =
    fun ([<InlineIfLambda>] vr) ([<InlineIfLambda>] sr) ->
      let n = Vector<int>.Count
      let mutable i = 0
      while i + n <= xs.Length do
        vr (Vector<int>(xs, i))
        i <- i + n
      while i < xs.Length do
        sr xs.[i]
        i <- i + 1

  // map: a vector op for the blocks + the matching scalar op for the tail.
  let inline map
      ([<InlineIfLambda>] vf : Vector<int> -> Vector<int>)
      ([<InlineIfLambda>] sf : int -> int)
      ([<InlineIfLambda>] ps : S) : S =
    fun ([<InlineIfLambda>] vr) ([<InlineIfLambda>] sr) ->
      ps (fun v -> vr (vf v)) (fun x -> sr (sf x))

  // Sink: sum (lanewise accumulate, one horizontal reduce at the end).
  let inline sum ([<InlineIfLambda>] ps : S) : int =
    let mutable vacc = Vector<int>.Zero
    let mutable sacc = 0
    ps (fun v -> vacc <- vacc + v) (fun x -> sacc <- sacc + x)
    Vector.Sum vacc + sacc

  // Sink: max (lanewise Vector.Max, then horizontal max over the lanes + tail).
  let inline max ([<InlineIfLambda>] ps : S) : int =
    let mutable vacc = Vector<int>(Int32.MinValue)
    let mutable sacc = Int32.MinValue
    ps (fun v -> vacc <- Vector.Max(vacc, v)) (fun x -> if x > sacc then sacc <- x)
    let mutable m = sacc
    for j in 0 .. Vector<int>.Count - 1 do
      if vacc.[j] > m then m <- vacc.[j]
    m

  // The fusion-preserving pipe (same trick as the scalar library).
  let inline (|>>) ([<InlineIfLambda>] v : _ -> _) ([<InlineIfLambda>] f : _ -> _) = f v

// Sum-of-squares via the SIMD push stream.
module Simd =
  open SimdStream
  let sumSq (xs : int[]) =
    ofArray xs
    |>> map (fun v -> v * v) (fun x -> x * x)
    |>> sum

// Sum-of-squares via the scalar fused PushStream (the |>> tier).
module Scalar =
  open PushStream6.PushStream
  let sumSq (xs : int[]) =
    ofArray xs
    |>> map (fun x -> x * x)
    |>> fold (+) 0

[<MemoryDiagnoser>]
type SimdBench() =

  [<Params(1000, 1000000)>]
  member val Length = 0 with get, set

  member val Xs : int[] = [||] with get, set

  [<GlobalSetup>]
  member this.Setup() =
    let r = Random 42
    this.Xs <- Array.init this.Length (fun _ -> r.Next(0, 4))

  // Plain hand-written scalar loop — the "what you'd type" baseline.
  [<Benchmark(Baseline = true)>]
  member this.HandScalar() =
    let xs = this.Xs
    let mutable s = 0
    for i in 0 .. xs.Length - 1 do
      s <- s + xs.[i] * xs.[i]
    s

  // Hand-written SIMD loop — the performance ceiling we're chasing.
  [<Benchmark>]
  member this.HandSimd() =
    let xs = this.Xs
    let n = Vector<int>.Count
    let mutable acc = Vector<int>.Zero
    let mutable i = 0
    while i + n <= xs.Length do
      let v = Vector<int>(xs, i)
      acc <- acc + v * v
      i <- i + n
    let mutable s = Vector.Sum acc
    while i < xs.Length do
      s <- s + xs.[i] * xs.[i]
      i <- i + 1
    s

  // Scalar fused PushStream6 — ties HandScalar (our earlier result).
  [<Benchmark>]
  member this.ScalarFused() = Scalar.sumSq this.Xs

  // The new SIMD push stream.
  [<Benchmark>]
  member this.SimdStream() = Simd.sumSq this.Xs

module Main =
  [<EntryPoint>]
  let main _ =
    // Correctness gate: a fast-but-wrong SIMD impl must fail here, not "win".
    let r = Random 1
    let xs = Array.init 1234 (fun _ -> r.Next(0, 100))
    let hand =
      let mutable s = 0
      for x in xs do s <- s + x * x
      s
    let simd = Simd.sumSq xs
    let scalar = Scalar.sumSq xs
    if hand <> simd || hand <> scalar then
      failwithf "Mismatch: hand=%d simd=%d scalar=%d" hand simd scalar
    printfn "Sanity OK (sum of squares = %d); Vector<int>.Count = %d" hand Vector<int>.Count
    BenchmarkRunner.Run<SimdBench>() |> ignore
    0
