/**
 * Jev router with three cost/quality modes. Jev classifies each new user prompt;
 * tool follow-ups and retries stay on the model chosen for that turn.
 *
 * Requires TYPESAFE_API_KEY and an OpenAI Codex login.
 * Usage: pi -e ./jev-router.ts --model jev/auto
 * Select jev/cheap or jev/max with /model or --model.
 */

import type { Message } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ModelRoute, ModelRouteRequest } from "@earendil-works/pi-coding-agent";

const PROVIDER = "openai-codex";
const SOL = "gpt-5.6-sol";
const TERRA = "gpt-5.6-terra";
const LUNA = "gpt-5.6-luna";

type Mode = "cheap" | "auto" | "max";
type Complexity = "simple" | "standard" | "complex";

interface JevState {
	model: string;
}

type JevRequest = ModelRouteRequest<JevState>;

const MODEL_BY_MODE: Record<Mode, Record<Complexity, string>> = {
	cheap: { simple: LUNA, standard: LUNA, complex: TERRA },
	auto: { simple: LUNA, standard: TERRA, complex: SOL },
	max: { simple: SOL, standard: SOL, complex: SOL },
};

function routeTo(request: JevRequest, ctx: ExtensionContext, id: string, state?: JevState): ModelRoute<JevState> {
	const model = ctx.modelRegistry.find(PROVIDER, id);
	if (!model) throw new Error(`Model ${PROVIDER}/${id} is not in the catalog`);
	return { model, thinkingLevel: request.thinkingLevel, state };
}

function lastUserText(messages: readonly Message[]): string {
	const content = messages.filter((message) => message.role === "user").at(-1)?.content ?? "";
	if (typeof content === "string") return content;
	return content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("\n");
}

async function classifyComplexity(request: JevRequest, ctx: ExtensionContext): Promise<Complexity> {
	const jev = ctx.modelRegistry.findOfType("classifier", "typesafe", "jev-latest");
	if (!jev) return "standard";
	const result = await ctx.modelRegistry.classify(
		jev,
		{
			state: { prompt: lastUserText(request.messages).slice(0, 16_000) },
			questions: {
				complexity: {
					type: "choice",
					instructions: "How demanding is the software engineering task requested in `prompt`?",
					criteria: {
						simple: "Short questions, explanations, or small mechanical edits",
						standard: "Ordinary features, fixes, reviews, or moderate debugging",
						complex: "Subtle design, cross-cutting changes, difficult debugging, or high-stakes review",
					},
				},
			},
		},
		{ signal: request.signal },
	);
	const answer = result.stopReason === "stop" ? result.answers.complexity : undefined;
	if (answer?.type !== "choice") return "standard";
	if (answer.choice === "simple" || answer.choice === "standard" || answer.choice === "complex") return answer.choice;
	return "standard";
}

export default function (pi: ExtensionAPI) {
	for (const mode of ["cheap", "auto", "max"] as const) {
		pi.registerVirtualModel<JevState>({
			provider: "jev",
			id: mode,
			name: `${mode[0].toUpperCase()}${mode.slice(1)} (Jev)`,
			thinkingLevels: ["low", "medium", "high", "xhigh"],
			contextWindow: 272_000,
			maxTokens: 128_000,
			async route(request, ctx) {
				if (request.reason === "direct") return routeTo(request, ctx, MODEL_BY_MODE[mode].standard);
				if (request.reason === "retry" && request.failed) {
					return { model: request.failed.model, thinkingLevel: request.failed.thinkingLevel ?? request.thinkingLevel };
				}
				if (request.reason === "continuation") {
					const id = request.state?.model;
					if (id) return routeTo(request, ctx, id);
					if (request.previous) {
						return { model: request.previous.model, thinkingLevel: request.previous.thinkingLevel ?? request.thinkingLevel };
					}
				}
				const complexity = mode === "max" ? "standard" : await classifyComplexity(request, ctx);
				const model = MODEL_BY_MODE[mode][complexity];
				return routeTo(request, ctx, model, { model });
			},
		});
	}
}
