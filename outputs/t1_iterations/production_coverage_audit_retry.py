"""Distinct retry preserves original failed audit and events."""
import production_coverage_audit as audit

audit.RUN = audit.HERE / 'private/production_coverage_audit_retry_01'
audit.PUBLIC = audit.HERE / 'PRODUCTION_COVERAGE_AUDIT_RETRY_RESULTS.json'

if __name__ == '__main__':
    audit.main()
