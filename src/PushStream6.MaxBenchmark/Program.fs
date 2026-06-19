namespace PushStream6.MaxBenchmark

open System
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Running

// System.Linq pipeline: Select(x/2) -> Where(even) -> Max
module SysLinq =
  open System.Linq
  let run (ints : int[]) =
    ints.Select(fun x -> x / 2).Where(fun x -> (x &&& 1) = 0).Max()

// Cistern.ValueLinq, both the "normal" (delegate) and "ugly" (struct IFunc) forms
module VLinq =
  open Cistern.ValueLinq

  let run (ints : int[]) : int =
    ints.Select(fun x -> x / 2).Where(fun x -> (x &&& 1) = 0).Max<_>()

  [<Struct>] type Halve = interface IFunc<int, int>  with member _.Invoke x = x / 2
  [<Struct>] type Even  = interface IFunc<int, bool> with member _.Invoke x = (x &&& 1) = 0

  let runFast (ints : int[]) : int =
    ints.Select(Halve(), 0).Where(Even()).Max<_>()

// F# push stream: ofArray -> map (x/2) -> filter even -> fold max
module PS =
  open PushStream6.PushStream

  // using standard |> (defeats lambda inlining)
  let runPipe (ints : int[]) =
    ints
    |> ofArray
    |> map    (fun x -> x / 2)
    |> filter (fun x -> (x &&& 1) = 0)
    |> fold   (fun a v -> if v > a then v else a) Int32.MinValue

  // using |>> (preserves InlineIfLambda fusion)
  let runFused (ints : int[]) =
    ofArray ints
    |>> map    (fun x -> x / 2)
    |>> filter (fun x -> (x &&& 1) = 0)
    |>> fold   (fun a v -> if v > a then v else a) Int32.MinValue

// Nessos.Streams (the established F# push/pull stream lib, pre-InlineIfLambda)
module Nessos =
  open Nessos.Streams
  let run (ints : int[]) =
    ints
    |> Stream.ofArray
    |> Stream.map    (fun x -> x / 2)
    |> Stream.filter (fun x -> (x &&& 1) = 0)
    |> Stream.fold   (fun a v -> if v > a then v else a) Int32.MinValue

[<MemoryDiagnoser>]
type MaxBench() =

  [<Params(1, 100, 1000000)>]
  member val Length = 0 with get, set

  member val Ints : int[] = [||] with get, set

  [<GlobalSetup>]
  member this.Setup() =
    let r = Random 42
    this.Ints <- Array.init this.Length (fun _ -> r.Next())

  [<Benchmark(Baseline = true)>]
  member this.Handcoded() =
    let arr = this.Ints
    let mutable m = Int32.MinValue
    for n in arr do
      let x = n / 2
      if (x &&& 1) = 0 then
        if x > m then m <- x
    m

  [<Benchmark>]
  member this.Linq() = SysLinq.run this.Ints

  [<Benchmark>]
  member this.ValueLinq() = VLinq.run this.Ints

  [<Benchmark>]
  member this.ValueLinqFast() = VLinq.runFast this.Ints

  [<Benchmark>]
  member this.PushStreamPipe() = PS.runPipe this.Ints

  [<Benchmark>]
  member this.PushStreamFused() = PS.runFused this.Ints

  [<Benchmark>]
  member this.NessosStreams() = Nessos.run this.Ints

module Main =
  [<EntryPoint>]
  let main _ =
    BenchmarkRunner.Run<MaxBench>() |> ignore
    0
