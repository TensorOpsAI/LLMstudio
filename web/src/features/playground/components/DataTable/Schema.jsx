import { z } from "zod";

export const taskSchema = z.object({
  id: z.number(),
  input: z.string(),
  output: z.string(),
  promptTokens: z.number(), // input tokens
  completionTokens: z.number(), // input tokens
  totalTokens: z.number(), // input tokens
  totalCost: z.number(), // cost
  timestamp: z.date(),
  model: z.string(),
  // parameters: z.union([z.number(), z.number(), z.number(), z.number(),z.number(),z.number()])
});
