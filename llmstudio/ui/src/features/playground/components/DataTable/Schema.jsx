import { z } from "zod";

export const taskSchema = z.object({
  id: z.number(),
  chatInput: z.string(),
  chatOutput: z.string(),
  inputTokens: z.number(), // input tokens
  outputTokens: z.number(), // input tokens
  totalTokens: z.number(), // input tokens
  cost: z.number(), // cost
  timestamp: z.date(),
  model: z.string(),
  // parameters: z.union([z.number(), z.number(), z.number(), z.number(),z.number(),z.number()])
});
