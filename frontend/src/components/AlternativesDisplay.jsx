import { memo } from 'react'
import {
  Sparkles, ArrowRight, CheckCircle2, AlertCircle,
  Pill, Replace, Shuffle
} from 'lucide-react'

const AlternativesDisplay = memo(function AlternativesDisplay({ alternatives }) {
  if (!alternatives) return null

  const {
    original_drug1,
    original_drug2,
    alternatives_for_drug1,
    alternatives_for_drug2,
    safe_combinations
  } = alternatives

  return (
    <section className="max-w-3xl mx-auto px-5 py-8">
      <div className="rounded-2xl p-6 sm:p-8 bg-[var(--bg-elevated)] border border-[var(--border)]">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-medical-500/10 border border-medical-500/15 flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-medical-400" />
          </div>
          <div>
            <h2 className="font-display text-xl font-bold text-[var(--text-primary)]">
              Safe Alternatives
            </h2>
            <p className="text-[var(--text-secondary)] text-sm">
              AI-recommended substitutes with no known interactions
            </p>
          </div>
        </div>

        {/* Alternatives for Drug 1 */}
        {alternatives_for_drug1 && alternatives_for_drug1.length > 0 && (
          <div className="mb-7">
            <div className="flex items-center gap-2.5 mb-3.5">
              <Replace className="w-4 h-4 text-medical-400" />
              <h3 className="font-semibold text-sm text-[var(--text-primary)]">
                Alternatives for <span className="text-medical-400">{original_drug1?.name}</span>
              </h3>
            </div>
            <div className="grid gap-2.5">
              {alternatives_for_drug1.map((alt, index) => (
                <AlternativeCard key={index} alternative={alt} />
              ))}
            </div>
          </div>
        )}

        {/* Alternatives for Drug 2 */}
        {alternatives_for_drug2 && alternatives_for_drug2.length > 0 && (
          <div className="mb-7">
            <div className="flex items-center gap-2.5 mb-3.5">
              <Replace className="w-4 h-4 text-medical-400" />
              <h3 className="font-semibold text-sm text-[var(--text-primary)]">
                Alternatives for <span className="text-medical-400">{original_drug2?.name}</span>
              </h3>
            </div>
            <div className="grid gap-2.5">
              {alternatives_for_drug2.map((alt, index) => (
                <AlternativeCard key={index} alternative={alt} />
              ))}
            </div>
          </div>
        )}

        {/* Safe Combinations */}
        {safe_combinations && safe_combinations.length > 0 && (
          <div className="pt-6 border-t border-[var(--border)]">
            <div className="flex items-center gap-2.5 mb-3.5">
              <Shuffle className="w-4 h-4 text-medical-400" />
              <h3 className="font-semibold text-sm text-[var(--text-primary)]">
                Recommended Safe Combinations
              </h3>
            </div>
            <div className="grid gap-2.5">
              {safe_combinations.slice(0, 5).map((combo, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3.5 bg-[var(--bg-primary)]/50 rounded-xl hover:bg-[var(--bg-primary)] transition-colors border border-[var(--border)]"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2">
                      <div className="p-1.5 bg-medical-500/8 rounded-lg">
                        <Pill className="w-3.5 h-3.5 text-medical-400" />
                      </div>
                      <span className="text-[var(--text-primary)] font-medium text-sm">{combo.drug1?.name}</span>
                    </div>
                    <span className="text-[var(--text-muted)] text-xs">+</span>
                    <div className="flex items-center gap-2">
                      <div className="p-1.5 bg-medical-500/8 rounded-lg">
                        <Pill className="w-3.5 h-3.5 text-medical-400" />
                      </div>
                      <span className="text-[var(--text-primary)] font-medium text-sm">{combo.drug2?.name}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                    <span className="text-green-400 text-xs font-medium">Safe</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="mt-7 p-4 bg-warning-500/6 border border-warning-500/12 rounded-xl">
          <div className="flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-warning-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-warning-500 dark:text-warning-400 font-medium text-sm mb-0.5">Important Notice</p>
              <p className="text-[var(--text-secondary)] text-xs leading-relaxed">
                These alternatives are suggestions based on similar therapeutic effects.
                Always consult with your healthcare provider before making changes to your medication.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
})

function AlternativeCard({ alternative }) {
  const { drug, similarity_score, reason, has_interaction_with_other } = alternative

  return (
    <div className="flex items-center justify-between p-3.5 bg-[var(--bg-primary)]/50 rounded-xl hover:bg-[var(--bg-primary)] transition-colors group border border-[var(--border)]">
      <div className="flex items-center gap-3.5">
        <div className="w-10 h-10 rounded-xl bg-medical-500/8 flex items-center justify-center">
          <Pill className="w-5 h-5 text-medical-400" />
        </div>
        <div>
          <p className="text-[var(--text-primary)] font-medium text-sm">{drug?.name}</p>
          {drug?.drug_class && (
            <p className="text-[var(--text-muted)] text-xs">{drug.drug_class}</p>
          )}
          {reason && (
            <p className="text-[var(--text-secondary)] text-xs mt-0.5">{reason}</p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3.5">
        {/* Similarity score */}
        <div className="text-right hidden sm:block">
          <p className="text-[var(--text-muted)] text-[10px] mb-1 uppercase tracking-wider">Similarity</p>
          <div className="flex items-center gap-2">
            <div className="w-14 h-1.5 bg-[var(--bg-elevated)] rounded-full overflow-hidden">
              <div
                className="h-full bg-medical-400 rounded-full"
                style={{ width: `${similarity_score * 100}%` }}
              />
            </div>
            <span className="text-medical-400 text-xs font-mono">
              {Math.round(similarity_score * 100)}%
            </span>
          </div>
        </div>

        {/* Interaction status */}
        <div className={`px-2.5 py-1 rounded-lg text-[10px] font-semibold uppercase tracking-wider border ${has_interaction_with_other
            ? 'bg-amber-500/8 text-amber-400 border-amber-500/15'
            : 'bg-green-500/8 text-green-400 border-green-500/15'
          }`}>
          {has_interaction_with_other ? 'Minor' : 'Safe'}
        </div>

        <ArrowRight className="w-4 h-4 text-[var(--text-muted)] group-hover:text-medical-400 group-hover:translate-x-0.5 transition-all" />
      </div>
    </div>
  )
}

export default AlternativesDisplay
