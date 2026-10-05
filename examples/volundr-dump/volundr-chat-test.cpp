// llama-volundr-chat-test: show what common/chat.cpp derives from a chat template (format, grammar triggers, preserved
// tokens, stops) and run the output parser on sample assistant outputs. Usage: llama-volundr-chat-test <template.jinja> [sample.txt ...]
#include "chat.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

static std::string slurp(const char * p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s template.jinja [samples...]\n", argv[0]);
        return 1;
    }
    auto tmpls = common_chat_templates_init(nullptr, slurp(argv[1]), "<|endoftext|>", "<|agens_end|>");

    for (int with_tools = 0; with_tools < 2; ++with_tools) {
        for (int think = 0; think < 2; ++think) {
            common_chat_templates_inputs in;
            common_chat_msg u;
            u.role    = "user";
            u.content = "What is the weather in Hong Kong?";
            in.messages.push_back(u);
            in.enable_thinking  = think;
            in.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            if (with_tools) {
                in.tools.push_back({ "get_weather", "Get the weather",
                    R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})" });
            }
            auto p = common_chat_templates_apply(tmpls.get(), in);
            printf("===== tools=%d think=%d format=%s grammar_lazy=%d supports_thinking=%d thinking_start=%s\n",
                with_tools, think, common_chat_format_name(p.format), p.grammar_lazy, p.supports_thinking, p.thinking_start_tag.c_str());
            for (auto & t : p.thinking_end_tags) printf("  thinking_end_tag: %s\n", t.c_str());
            for (auto & t : p.grammar_triggers)  printf("  trigger type=%d value=[%s]\n", (int) t.type, t.value.c_str());
            for (auto & t : p.preserved_tokens)  printf("  preserved: [%s]\n", t.c_str());
            for (auto & t : p.additional_stops)  printf("  stop: [%s]\n", t.c_str());
            printf("  generation_prompt: [%s]\n", p.generation_prompt.c_str());
            printf("  prompt tail: [%s]\n", p.prompt.substr(p.prompt.size() > 300 ? p.prompt.size() - 300 : 0).c_str());
            if (with_tools) printf("  grammar (head): %s\n", p.grammar.substr(0, 1500).c_str());

            for (int i = 2; i < argc; ++i) {
                common_chat_parser_params pp(p);
                pp.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
                if (!p.parser.empty()) {
                    pp.parser.load(p.parser);
                }
                const std::string s = slurp(argv[i]);
                try {
                    auto m = common_chat_parse(s, false, pp);
                    printf("  -- parse %s: reasoning=[%s] content=[%s] n_tool_calls=%zu\n", argv[i], m.reasoning_content.c_str(), m.content.c_str(), m.tool_calls.size());
                    for (auto & tc : m.tool_calls) printf("     call %s %s\n", tc.name.c_str(), tc.arguments.c_str());
                } catch (const std::exception & e) {
                    printf("  -- parse %s: EXCEPTION %s\n", argv[i], e.what());
                }
            }
        }
    }
    return 0;
}
